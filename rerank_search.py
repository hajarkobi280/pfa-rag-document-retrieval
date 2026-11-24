"""Reclassement question-passage en franÃ§ais, uniquement en local."""

from rag_observability import measure
import argparse

from collections import defaultdict

import json

import os

from pathlib import Path



from hybrid_search import HybridSearch

import numpy as np

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification



ROOT=Path(__file__).resolve().parent

MODEL_DIR=ROOT/'models/reranker-fr'



# --- RÃ©glages de vitesse et de pertinence ------------------------------------

MAX_LEN=384        # fenÃªtre du reclasseur (avant : 512)

POOL_SIZE=12       # reduit pour accelerer CPU       # candidats reclassÃ©s (avant : 80)

BATCH_SIZE=4       # lot reduit memoire

MIN_SCORE=float(os.environ["RAG_MIN_SCORE"]) if os.environ.get("RAG_MIN_SCORE") else None     # seuil du reclasseur (logit brut) ; None = dÃ©sactivÃ©. Ã€ calibrer.

SEM_MIN=float(os.environ["RAG_SEM_MIN"]) if os.environ.get("RAG_SEM_MIN") else None       # seuil sÃ©mantique E5 pour le refus immÃ©diat ; None = dÃ©sactivÃ©. Ã€ calibrer.

# -----------------------------------------------------------------------------



torch.set_num_threads(os.cpu_count() or 4)





def select_diverse(rows, top_k, max_per_document):

    selected,counts,seen=[],defaultdict(int),set()

    for row in rows:

        if row['chunk_id'] in seen or counts[row['doc_id']]>=max_per_document:

            continue

        selected.append(row)

        seen.add(row['chunk_id'])

        counts[row['doc_id']]+=1

        if len(selected)==top_k:

            break

    return selected





class RerankedSearch:

    def __init__(self):

        if not (MODEL_DIR/'download_complete.json').exists() or not (MODEL_DIR/'model.safetensors').is_file():

            raise ValueError('ModÃ¨le absent : exÃ©cuter py download_reranker.py')

        self.engine=HybridSearch()

        self.raw={r['chunk_id']:r for r in map(json.loads,(ROOT/'data/service_public_clean.jsonl').open(encoding='utf-8'))}

        self.by_doc=defaultdict(list)

        self.ids=list(self.raw)

        for i,cid in enumerate(self.ids):

            self.by_doc[self.raw[cid]['doc_id']].append(i)

        # Expansion dans les fiches retrouvÃ©es, d'aprÃ¨s le corps utile et non

        # l'introduction rÃ©pÃ©tÃ©e. RÃ©utilise le vocabulaire lexical existant.

        self.body_matrix=self.engine.vectorizer.transform(self.scoring_text(self.raw[cid]) for cid in self.ids)

        print('Chargement du reclasseur franÃ§ais...',flush=True)

        self.tokenizer=AutoTokenizer.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False)

        self.model=AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR),local_files_only=True,trust_remote_code=False,use_safetensors=True)

        self.model.eval()

        # XLM-RoBERTa rÃ©serve deux positions ; ce modÃ¨le dispose de 514 positions.

        if self.model.config.model_type != 'xlm-roberta' or self.model.config.max_position_embeddings < MAX_LEN+2:

            raise ValueError('Le reclasseur ne prend pas en charge les fenÃªtres demandÃ©es.')

        self.tokenizer.model_max_length=MAX_LEN

        # GPU si disponible, sinon quantification int8 sur CPU (plus rapide).

        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        if self.device=='cuda':

            self.model.to('cuda')

        else:

            try:

                self.model=torch.quantization.quantize_dynamic(self.model,{torch.nn.Linear},dtype=torch.qint8)

            except Exception as exc:

                print(f'Quantification int8 ignorÃ©e : {exc}',flush=True)

        self.metadata=json.loads((MODEL_DIR/'download_complete.json').read_text(encoding='utf-8'))



    @staticmethod

    def scoring_text(row):

        body=row.get('text') or row['chunk_text']

        context=' / '.join(row.get('context',[]))

        return row['title']+'\n'+context+'\n'+body



    def windows(self, question, row):

        # Le titre et le contexte restent prÃ©sents dans chaque fenÃªtre.

        header=row['title']+'\n'+' / '.join(row.get('context',[]))

        header_ids=self.tokenizer.encode(header,add_special_tokens=False)[:64]

        header=self.tokenizer.decode(header_ids,skip_special_tokens=True)

        question_length=len(self.tokenizer.encode(question,add_special_tokens=False))

        if question_length>128:

            raise ValueError('Question trop longue : utiliser au maximum 128 tokens.')

        body=row.get('text') or row['chunk_text']

        offsets=self.tokenizer(body,add_special_tokens=False,return_offsets_mapping=True,truncation=False,verbose=False)['offset_mapping']

        budget=MAX_LEN-question_length-len(header_ids)-12

        if budget<64:

            raise ValueError('Pas assez de place pour le passage.')

        fragments=[]

        for start in range(0,len(offsets),budget-48):

            end=min(len(offsets),start+budget)

            fragment=header+'\n'+body[offsets[start][0]:offsets[end-1][1]]

            # Aucune troncature silencieuse du passage.

            length=len(self.tokenizer(question,fragment,add_special_tokens=True,truncation=False,verbose=False)['input_ids'])

            if length>MAX_LEN:

                raise ValueError('FenÃªtre trop longue pour le reclasseur.')

            fragments.append(fragment)

            if end==len(offsets):

                break

        return fragments



    def search(self,question,top_k=5,pool_size=POOL_SIZE,max_per_document=3):

        if not question.strip() or min(top_k,pool_size,max_per_document)<1 or top_k>pool_size:

            raise ValueError('Question ou limites invalides')

        base=self.engine.search(question,top_k=top_k,candidates=max(40,pool_size),include_candidates=True)

        original=base.pop('candidate_pool')

        # Refus immÃ©diat : rien de lexical et score sÃ©mantique trop bas -> question hors base.

        if SEM_MIN is not None and base['semantic_top_score']<SEM_MIN and base['lexical_hits']==0:

            base['reranked']=[]

            base['hybrid_relaxed']=[]

            return base

        with measure('reranking'):
            pool={r['chunk_id']:r for r in original[:pool_size//2]}

            # Trouver aussi des passages procÃ©duraux sous-classÃ©s dans les 6 premiÃ¨res fiches.

            docs=list(dict.fromkeys(r['doc_id'] for r in original))[:6]

            scores=(self.body_matrix @ self.engine.vectorizer.transform([question]).T).toarray().ravel()

            doc_lists=[]

            for doc in docs:

                indices=sorted(self.by_doc[doc],key=lambda i:(-float(scores[i]),self.ids[i]))

                doc_lists.append(indices)

            for position in range(max(map(len,doc_lists),default=0)):

                for indices in doc_lists:

                    if position<len(indices):

                        cid=self.ids[indices[position]]

                        if cid not in pool and len(pool)<pool_size:

                            pool[cid]={**self.engine.by_id[cid],'candidate_origin':'document_expansion'}

            for row in original:

                if len(pool)>=pool_size:

                    break

                pool.setdefault(row['chunk_id'],row)

            candidates=list(pool.values())

            pairs,owners,fragments=[],[],[]

            for i,row in enumerate(candidates):

                for fragment in self.windows(question,self.raw[row['chunk_id']]):

                    pairs.append((question,fragment))

                    owners.append(i)

                    fragments.append(fragment)

            best=np.full(len(candidates),-np.inf,dtype=np.float32)

            best_window={}

            # Tri par longueur : moins de padding gaspillÃ© dans chaque lot.

            order=sorted(range(len(pairs)),key=lambda j:len(pairs[j][1]))

            with torch.inference_mode():

                for start in range(0,len(order),BATCH_SIZE):

                    idx=order[start:start+BATCH_SIZE]

                    batch=[pairs[j] for j in idx]

                    inputs=self.tokenizer([p[0] for p in batch],[p[1] for p in batch],padding=True,truncation=False,return_tensors='pt')

                    if self.device=='cuda':

                        inputs={k:v.to('cuda') for k,v in inputs.items()}

                    values=self.model(**inputs).logits.reshape(-1).float().cpu().numpy()

                    if not np.isfinite(values).all():

                        raise ValueError('Scores du reclasseur invalides')

                    for j,value in zip(idx,values):

                        owner=owners[j]

                        if value>best[owner]:

                            best[owner]=value

                            best_window[owner]=fragments[j]

            ranked=[]

            for i in np.argsort(-best,kind='stable'):

                row=dict(candidates[int(i)])

                if 'score' in row:

                    row['rrf_score']=row.pop('score')

                row['rerank_score']=float(best[i])

                row['scored_excerpt']=best_window[int(i)]

                ranked.append(row)

            # Seuil de pertinence : une liste vide dÃ©clenche un refus immÃ©diat, sans appel LLM.

            if MIN_SCORE is not None:

                ranked=[r for r in ranked if r['rerank_score']>=MIN_SCORE]

            base['hybrid_relaxed']=select_diverse(original,top_k,max_per_document)

            base['reranked']=select_diverse(ranked,top_k,max_per_document)

            base['rerank_metadata']={'model':self.metadata['model'],'revision':self.metadata['revision'],

                'pool_size':len(candidates),'windows_scored':len(pairs),'max_per_document':max_per_document,

                'candidate_chunk_ids':[r['chunk_id'] for r in candidates],

                'scoring':'maximum raw relevance logit over windows; not probability or guarantee of answerability',

                'input':'title + context + body; repeated introduction omitted when body exists'}

        return base





def main():

    parser=argparse.ArgumentParser(description=__doc__)

    parser.add_argument('question')

    parser.add_argument('--top-k',type=int,default=5)

    parser.add_argument('--pool-size',type=int,default=POOL_SIZE)

    parser.add_argument('--max-per-document',type=int,default=3)

    parser.add_argument('--compare',action='store_true')

    args=parser.parse_args()

    if min(args.top_k,args.pool_size,args.max_per_document)<1 or args.top_k>args.pool_size:

        parser.error('Limites positives requises ; pool-size >= top-k')

    result=RerankedSearch().search(args.question,args.top_k,args.pool_size,args.max_per_document)

    print(json.dumps(result if args.compare else result['reranked'],ensure_ascii=False,indent=2))





if __name__=='__main__':

    main()

