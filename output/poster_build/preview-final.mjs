import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('D:/documents/DSBD/PFA CODE/output/poster/Poster_RAG_ICAIS26_KOBI_HAJAR.pptx'));
const dir='D:/documents/DSBD/PFA CODE/output/poster_build';
await fs.writeFile(dir+'/final.png',new Uint8Array(await (await p.export({slide:p.slides.items[0],format:'png',scale:.42})).arrayBuffer()));
await fs.writeFile(dir+'/final-inspection.txt',(await p.inspect({kind:'slide,textbox,shape,image,layout',maxChars:100000})).ndjson);
await fs.writeFile(dir+'/final.layout.json',await (await p.slides.items[0].export({format:'layout'})).text());
console.log('slides',p.slides.items.length);

