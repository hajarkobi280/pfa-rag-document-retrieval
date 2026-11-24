"""Figures reproduites à partir des mesures locales, sans valeurs estimées."""
import json
from pathlib import Path
import statistics
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pfa_figures'
OUT.mkdir(parents=True, exist_ok=True)
before = json.loads((ROOT/'data/benchmark_before.json').read_text(encoding='utf-8'))
report = json.loads((ROOT/'data/benchmark_models.json').read_text(encoding='utf-8'))
rows = [r for r in report['models'] if r.get('available')]
times = [r['seconds'] for r in before]
medians = [statistics.median(times)] + [r['median_seconds'] for r in rows]
p95 = [sorted(times)[math.ceil(.95*len(times))-1]] + [r['p95_seconds'] for r in rows]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
labels = ['Avant', 'Après', 'Après + revue']
fig, ax = plt.subplots(figsize=(10,5.5))
x = np.arange(3)
for offset,values,label,color in [(-.19,medians,'Médiane','#2155a3'),(.19,p95,'p95','#e69a35')]:
    bars=ax.bar(x+offset,values,.36,label=label,color=color)
    ax.bar_label(bars,labels=[f'{v:.2f} s'.replace('.',',') for v in values],padding=5)
ax.set(xticks=x,xticklabels=labels,ylabel='Durée (secondes)',ylim=(0,46),title='Temps de sélection par le LLM et de validation — qwen3:1.7b')
ax.legend(frameon=False); ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.text(.5,.025,'9 questions par configuration • recherche exclue • un seul passage, conditions non uniformisées\nComparaison exploratoire ; avec 9 valeurs, le p95 correspond au maximum.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.10,1,1))
for ext in ['png','svg']:fig.savefig(OUT/f'latence.{ext}',dpi=180)
plt.close(fig)

fig, axes=plt.subplots(1,3,figsize=(12,5))
series=[([9,9,9],9,'JSON conforme au schéma'),([5,6,6],6,'Preuves attendues sans parasite'),([0,0,0],3,'Abstentions correctes')]
for ax,(values,total,title) in zip(axes,series):
    bars=ax.bar(labels,values,color=['#8494aa','#2155a3','#e69a35'])
    ax.bar_label(bars,labels=[f'{v}/{total}' for v in values],padding=5)
    ax.set(title=title,ylim=(0,total*1.25),ylabel='Cas réussis')
    ax.tick_params(axis='x',labelrotation=15)
    ax.set_yticks(range(total+1));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Qualité sur le jeu fixe — génération isolée, sans routage applicatif',fontsize=14)
fig.text(.5,.025,'9 questions : 6 répondables + 3 hors base. Ces résultats ne mesurent pas une précision générale.\nUne citation exacte ne garantit pas sa pertinence ; la revue ne corrige pas ici les mauvaises abstentions.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.12,1,.94))
for ext in ['png','svg']:fig.savefig(OUT/f'qualite.{ext}',dpi=180)
plt.close(fig)
print(OUT)
