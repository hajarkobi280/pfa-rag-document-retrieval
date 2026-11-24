import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
process.on('uncaughtException',e=>{console.error(e.message);process.exit(1);});
const ROOT='D:/documents/DSBD/PFA CODE';
const BUILD=ROOT+'/output/poster_build';
const SKILL='C:/Users/Hajar/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const utils=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
const p=await PresentationFile.importPptx(await FileBlob.load('D:/ICAIS26_Poster_Template_A0.pptx'));
const s=p.slides.items[0];
const objects=(await p.inspect({kind:'textbox,shape,image',maxChars:150000})).ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x));
const navy='#0B1E3F',blue='#2155A3',green='#16846A',muted='#5C6576',gold='#E69A35';
function find(t){const x=objects.find(x=>x.text===t);if(!x)throw Error('Missing '+t);return p.resolve(x.id);}
function edit(old,text,size){const sh=find(old);sh.text=text;if(size)sh.text.style={typeface:'Arial',fontSize:size};return sh;}
function tx(text,x,y,w,h,size=28,bold=false,color=navy,align='left'){
 const sh=s.shapes.add({geometry:'textbox',name:text.slice(0,55),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=text;sh.text.style={typeface:'Arial',fontSize:size,bold,color,alignment:align};sh.text.insets=0;return sh;
}
function eraseWhere(fn){p.delete(objects.filter(x=>x.position&&fn(x)).map(x=>x.id));}
function box(text,x,y,w,h,fill='#EAF1FA',size=28){
 const sh=s.shapes.add({geometry:'roundRect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'#BCC7DA',width:1.4}});
 sh.text=text;sh.text.style={typeface:'Arial',fontSize:size,bold:true,color:navy,alignment:'center',verticalAlignment:'middle'};sh.text.insets=12;return sh;
}
function down(x,y,h=25){s.shapes.add({geometry:'downArrow',position:{left:x-13,top:y,width:26,height:h},fill:blue,line:{fill:'none',width:0}});}

const title=edit('[YOUR RESEARCH POSTER TITLE]','Assistant RAG local pour\nles démarches administratives',70);
title.position={left:1005,top:305,width:2090,height:190};
edit('First Author¹ · Second Author² · Third Author¹','Réalisé par KOBI HAJAR',32);
const affiliation=edit('¹ Institution / Laboratory, City, Country   ·   ² Institution, City, Country','Master Data Science & Big Data\nFSBM, Université Hassan II de Casablanca, Maroc',27);
affiliation.position={left:1008,top:661,width:2030,height:76};
tx('Encadrant : Abdelaziz Marzak',1008,586,2000,37,28,false,'#E8EEF8');
tx('Co-encadrantes : Rawane Azeroual · Lamharmech Chaimaa',1008,624,2050,35,27,false,'#E8EEF8');
p.delete([objects.find(x=>x.text==='PAPER ID: XXX').id,objects.find(x=>x.name==='Rounded Rectangle 16').id]);

edit('INTRODUCTION','PROBLÉMATIQUE');
edit('OBJECTIVES','OBJECTIFS');
edit('DATA / MATERIALS','DONNÉES ET ENVIRONNEMENT');
edit('KEYWORDS','ÉTAT DE L’ART');
edit('METHODOLOGY','MÉTHODOLOGIE');
edit('RESULTS','RÉSULTATS EXPÉRIMENTAUX');
edit('KEY FINDINGS','APPORTS DU PROTOTYPE');
edit('DISCUSSION','DISCUSSION ET LIMITES');
edit('CONCLUSION','CONCLUSION');
edit('REFERENCES & ACKNOWLEDGEMENTS','RÉFÉRENCES',31);

const intro=objects.find(x=>x.text?.startsWith('Briefly introduce'));
p.resolve(intro.id).text='Les informations administratives sont dispersées entre fiches officielles et PDF personnels. Une réponse fluide peut confondre deux faits, même lorsque les sources sont disponibles.\n\nComment retrouver une preuve pertinente et fournir une réponse vérifiable sur une machine de 8 Go de RAM ?\n\nCas observé : confusion entre loyer et dépôt de garantie, tous deux de 450 euros.';
p.resolve(intro.id).text.style={typeface:'Arial',fontSize:29,color:navy};
const objective=objects.find(x=>x.text?.startsWith('• Main objective'));
p.resolve(objective.id).text='Retrouver les informations dans les PDF et la base administrative.\n\nAdapter les preuves à la question et afficher des citations exactes.\n\nMesurer la qualité et les temps de traitement en préservant le fonctionnement local.';
p.resolve(objective.id).text.style={typeface:'Arial',fontSize:29,color:navy};

// Replace content placeholders only; preserve the original panels and their headers.
eraseWhere(o=>o.position.left>=140&&o.position.left<1050&&o.position.top>=2090&&o.position.top<2640);
tx('35 850 passages administratifs\n47 415 fenêtres vectorisées\nEmbeddings E5 de dimension 384',148,2118,860,175,34,true);
tx('Base Service-Public locale et PDF personnels.\nFenêtres de 440 tokens, chevauchement de 60.\nPDF : 1 200 caractères, chevauchement de 200.',148,2320,860,150,28);
tx('Windows, Python 3.14, Flask et Ollama 0.40.11.\nqwen3:1.7b, contexte de 4 096 tokens.\nModèles préentraînés, sans fine-tuning.',148,2490,860,125,27);

eraseWhere(o=>o.position.left>=140&&o.position.left<1050&&o.position.top>=2820&&o.position.top<4190);
const lit=[
 ['Recherche augmentée [1]','Le RAG associe recherche documentaire et modèle de langage. Les sources améliorent la traçabilité, sans garantir l’absence d’erreurs.'],
 ['Recherche lexicale et sémantique [2]','TF-IDF retrouve les termes précis. E5 multilingue rapproche les formulations par le sens. Les deux approches apportent des signaux complémentaires.'],
 ['Fusion et reclassement [3, 4]','RRF combine les rangs des recherches. Le cross-encoder français réévalue les couples question–passage avant la sélection des preuves.'],
 ['Sortie structurée et extraction [5]','Le JSON facilite la validation informatique. L’application affiche les citations sources plutôt que la reformulation libre du modèle.']
];
for(let i=0;i<lit.length;i++){tx(lit[i][0],148,2860+i*225,860,48,32,true);tx(lit[i][1],148,2918+i*225,850,152,29);}
tx('POSITIONNEMENT DU PROJET',148,3800,850,45,30,true);
tx('Adapter une architecture RAG existante aux démarches administratives, puis évaluer la sélection des preuves et l’abstention sur une machine aux ressources limitées.',148,3860,850,170,29);

const methodIntro=objects.find(x=>x.text?.startsWith('Show the workflow'));
p.resolve(methodIntro.id).text='Recherche et inférence locales. Internet intervient seulement si le parcours local ne suffit pas et si la demande l’autorise.';
p.resolve(methodIntro.id).text.style={typeface:'Arial',fontSize:27,color:navy};
eraseWhere(o=>o.position.left>=1150&&o.position.left<2045&&o.position.top>=1080&&o.position.top<1870);
box('Base Service-Public\nTF-IDF + E5',1164,1080,390,100);
box('PDF personnels\nExtraction + E5',1590,1080,390,100);
down(1360,1185);down(1780,1185);
box('Question et recherche locale\nFusion RRF + reclasseur pour la base',1164,1220,816,100);
down(1572,1325);
box('Contrôle du sujet et tri des preuves\n6 unités au maximum',1164,1360,816,90);
down(1572,1455);
box('Ollama / Qwen3 : sélection des preuves\nJSON puis contrôle des citations',1164,1490,816,95);
down(1572,1590);
box('Citations exactes avec références [S1]\nou message d’insuffisance',1164,1625,816,90,'#EAF6F0');
tx('Si nécessaire : recherche web, puis nouvelle validation.\nDemande explicite sur un PDF : parcours local uniquement.\nAperçu PDF : extraits directs, sans appel au LLM.',1164,1740,816,105,25);

// Results area: native charts and complete experimental qualifiers.
eraseWhere(o=>o.position.left>=1150&&o.position.left<2045&&o.position.top>=2080&&o.position.top<4185);
tx('A. Temps de sélection et de validation',1162,2090,850,52,32,true);
function chart(title,cats,series,y,max,unit,fmt='0'){
 const c=s.charts.add('bar',{position:{left:1160,top:y,width:860,height:400},title,
  titleTextStyle:{fontSize:26,fill:navy},categories:cats,series,
  barOptions:{direction:'column',grouping:'clustered',gapWidth:100},hasLegend:true,
  legend:{position:'bottom',textStyle:{fontSize:23,fill:navy}},
  xAxis:{textStyle:{fontSize:24,fill:navy}},
  yAxis:{min:0,max,majorUnit:unit,textStyle:{fontSize:22,fill:muted},majorGridlines:{fill:'#E0E4EA',width:1},numberFormatCode:fmt},
  dataLabels:{showValue:true,position:'outEnd',numberFormatCode:fmt,textStyle:{fontSize:24,fill:navy,bold:true}},
  chartFill:'#FFFFFF',chartLine:{fill:'none',width:0}});
 utils.applyPresentationChartFont(c,{fontFamily:'Arial'});return c;
}
chart('Durée en secondes, recherche exclue',['Avant','Après','Avec revue'],[
 {name:'Médiane',values:[21.232,11.249,20.791],fill:blue},
 {name:'p95',values:[38.342,22.189,36.909],fill:gold}],2160,45,10,'0.00');
tx('9 questions, un passage par variante. Conditions de charge\nnon uniformisées. Le p95 correspond ici au maximum.\nLa revue ajoute du temps sans améliorer l’abstention.',1162,2600,845,115,26);
tx('B. Effet du contrôle de pertinence',1162,2770,850,52,32,true);
chart('Cas réussis sur le même jeu de 9 questions',['Sans contrôle','Avec contrôle'],[
 {name:'Abstentions / 3',values:[0,3],fill:green},
 {name:'Preuves pertinentes / 6',values:[6,6],fill:blue}],2840,7,1);
tx('Les 3 abstentions proviennent du filtre avant le LLM.\nLes 6 cas répondables conservent les preuves attendues.\nLe modèle lui-même n’a pas changé.',1162,3270,845,115,27);
tx('C. Validation sur 16 nouveaux exemples',1162,3440,850,52,31,true);
chart('Évaluation du filtre lexical seul',['Acceptées à raison','Rejetées à raison'],[
 {name:'Cas corrects / 8',values:[8,7],fill:green}],3510,9,2);
tx('0 faux refus sur 8 questions répondables.\n1 fausse acceptation sur 8 questions non répondables.\nCes petits jeux ne mesurent pas une précision générale.',1162,3950,845,125,27);

edit('Finding one','Citations vérifiables',30);
edit('State the result in one sentence.','Le texte affiché est présent dans les passages sources.',25);
edit('Finding two','Abstention avant génération',30);
edit('Use a number when relevant.','Les 3 cas hors base du jeu fixe sont refusés sans appel LLM.',25);
edit('Finding three','Fonctionnement sur 8 Go',30);
edit('Explain the practical meaning.','Recherche et modèles locaux, extension Internet explicite.',25);

const discussion=objects.find(x=>x.text?.startsWith('Interpret the results'));
p.resolve(discussion.id).text='Une citation exacte peut rester hors sujet. Le filtre lexical conserve les 8 bonnes réponses du jeu complémentaire, mais accepte encore 1 preuve incorrecte sur 8.\n\nLe mot « date » peut rapprocher une date de mariage et une échéance de dépôt. Les synonymes peuvent aussi échapper au contrôle.\n\nLes seuils sémantiques restent inchangés. La revue et le chemin rapide restent désactivés. Les modèles 4B n’étaient pas installés lors de l’évaluation.';
p.resolve(discussion.id).text.style={typeface:'Arial',fontSize:28,color:navy};
const conclusion=objects.find(x=>x.text?.startsWith('Summarize the contribution'));
p.resolve(conclusion.id).text='Le prototype associe recherche hybride, preuves adaptées à la question et citations exactes. Le contrôle du sujet améliore l’abstention sur le jeu fixe.\n\nLa généralisation exige davantage de questions, une calibration sémantique et une comparaison des modèles 4B.';
p.resolve(conclusion.id).text.style={typeface:'Arial',fontSize:29,color:navy};

const refs=objects.find(x=>x.text?.startsWith('[1] Author et al.'));
p.resolve(refs.id).text='[1] Lewis P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.\narxiv.org/abs/2005.11401\n\n[2] Wang L. et al. (2024). Multilingual E5 Text Embeddings: A Technical Report.\narxiv.org/abs/2402.05672\n\n[3] Cormack G. V., Clarke C. L. A., Büttcher S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. SIGIR, 758–759.\n\n[4] Louis A. Cross-encoder mMiniLMv2-L6, mMARCO français. Fiche du modèle Hugging Face.\nhuggingface.co/antoinelouis/\ncrossencoder-mMiniLMv2-L6-mmarcoFR\n\n[5] Ollama. Structured Outputs. Documentation officielle.\ndocs.ollama.com/capabilities/structured-outputs\n\nSOURCE DOCUMENTAIRE\nService-Public, copie locale indexée.\nwww.service-public.gouv.fr\n\nMESURES DU PROJET\nBenchmarks et tests locaux des 26–27 octobre 2026.\n90 tests Python réussis lors de la dernière exécution hors tests lents.\n\nLes références complètes et les sources de données figurent également dans les notes du poster.';
p.resolve(refs.id).text.style={typeface:'Arial',fontSize:27,color:navy};

s.speakerNotes.text='Contenu : D:/pfa.docx. Mise en page : D:/ICAIS26_Poster_Template_A0.pptx. Auteurs fournis par l’utilisatrice.\n'+
'Résultats : data/benchmark_before.json, data/benchmark_models.json (avant contrôle lexical), data/benchmark_topic_guard.json, data/topic_calibration.json.\n'+
'Figures sources : output/pfa_figures/latence.png, qualite.png, qualite_controle_pertinence.png. Graphiques reconstruits comme objets natifs modifiables.\n'+
'Le modèle seul s’abstenait 0/3, y compris avec revue. Le filtre applicatif obtient 3/3 sur les mêmes cas sans appeler le modèle, avec 6/6 cas répondables corrects. Validation distincte du filtre : 8 positifs conservés, 7 négatifs rejetés sur 8. Aucun gain général garanti.\n'+
'Références : https://arxiv.org/abs/2005.11401 ; https://arxiv.org/abs/2402.05672 ; https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf ; https://huggingface.co/antoinelouis/crossencoder-mMiniLMv2-L6-mmarcoFR ; https://docs.ollama.com/capabilities/structured-outputs ; https://www.service-public.gouv.fr/ .';
await fs.writeFile(BUILD+'/poster.png',new Uint8Array(await (await p.export({slide:s,format:'png',scale:.5})).arrayBuffer()));
await fs.writeFile(BUILD+'/poster.layout.json',await (await s.export({format:'layout'})).text());
await (await PresentationFile.exportPptx(p)).save(BUILD+'/candidate.pptx');
const hash=crypto.createHash('sha256').update(await fs.readFile('D:/ICAIS26_Poster_Template_A0.pptx')).digest('hex');
const result=await utils.finalizePresentation({
 workspaceDir:ROOT,candidatePath:BUILD+'/candidate.pptx',finalPath:ROOT+'/output/poster/Poster_RAG_ICAIS26_KOBI_HAJAR.pptx',
 pythonExecutable:'C:/Users/Hajar/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','30275213,42803763','--validate-heading-fit'],
 explicitTotalSlideCount:1,requiredNativeChartOwnerSlides:[1],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'reference',families:['Arial'],referencePath:'D:/ICAIS26_Poster_Template_A0.pptx',referenceSha256:hash},
 verifyArtifactToolImport:true,receiptPath:BUILD+'/validation.json'});
console.log(JSON.stringify(result));

