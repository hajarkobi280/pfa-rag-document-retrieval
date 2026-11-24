import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('D:/ICAIS26_Poster_Template_A0.pptx'));
const dir='D:/documents/DSBD/PFA CODE/output/poster_build';
await fs.writeFile(dir+'/template.png',new Uint8Array(await (await p.export({slide:p.slides.items[0],format:'png',scale:.42})).arrayBuffer()));
await fs.writeFile(dir+'/inspection.txt',(await p.inspect({kind:'slide,textbox,shape,image,layout',maxChars:100000})).ndjson);
await fs.writeFile(dir+'/template.layout.json',await (await p.slides.items[0].export({format:'layout'})).text());
console.log('slides',p.slides.items.length);
