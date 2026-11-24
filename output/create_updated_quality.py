"""Compare historical measurements with the new application topic guard."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'output/pfa_figures'
out.mkdir(parents=True, exist_ok=True)
old = json.loads((ROOT/'data/benchmark_models.json').read_text(encoding='utf-8'))
new = json.loads((ROOT/'data/benchmark_topic_guard.json').read_text(encoding='utf-8'))
standard = next(r for r in old['models'] if r.get('variant') == 'standard' and r.get('available'))
guard = next(r for r in new['models'] if r.get('available'))
def counts(entry, key):
    values = [r[key] for r in entry['results'] if r.get(key) is not None]
    return sum(values), len(values)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                     'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1,2,figsize=(11,6))
labels = ['Sans contrôle\nde sujet', 'Avec contrôle\nde sujet']
for ax, key, title in zip(axes,
        ['abstention_correct','evidence_correct'],
        ['Abstentions correctes', 'Preuves pertinentes, sans parasite']):
    results = [counts(entry,key) for entry in (standard,guard)]
    values = [r[0] for r in results]
    maximum = max(r[1] for r in results)
    bars = ax.bar(labels,values,color=['#94a3b8','#16846a'],width=.55)
    ax.bar_label(bars,labels=[f'{v}/{n}' for v,n in results],padding=9,fontsize=15,fontweight='bold')
    ax.set_ylim(0,maximum*1.25)
    ax.set_yticks(range(maximum+1))
    ax.set_ylabel('Cas réussis')
    ax.set_title(title,pad=15,fontsize=14)
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
fig.suptitle('Effet du nouveau contrôle de pertinence',fontsize=19,fontweight='bold',y=.97)
fig.text(.5,.89,'qwen3:1.7b • même jeu fixe de 9 questions',ha='center',fontsize=12,color='#475569')
fig.text(.5,.075,'Les 3 abstentions sont déclenchées par le contrôle avant le LLM, sans appel au modèle.\n'
         'Les 6 questions répondables restent réussies. Résultat exploratoire, pas une précision générale.',
         ha='center',fontsize=10,color='#334155',linespacing=1.6)
fig.tight_layout(rect=(0,.18,1,.86),w_pad=3)
for ext in ('png','svg'):
    fig.savefig(out/f'qualite_controle_pertinence.{ext}',dpi=190,facecolor='white')
plt.close(fig)
print(out/'qualite_controle_pertinence.png')
