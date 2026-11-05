"""Legacy evaluation of the retired MFCC+DTW speaker matcher (kept for reference).

The production algorithm is now MFCC/LPC statistics + RBF SVM; use
scripts/evaluate_speaker_svm.py for current results. This script reproduces
the old DTW baseline by scoring each test recording against the training
MFCC templates directly.
"""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dsp_speech.dtw import dtw_distance
from dsp_speech.io_audio import load_wav
from dsp_speech.recognize import prepare_utterance
from dsp_speech.speaker_recognition import load_speaker_templates

templates,names=load_speaker_templates(ROOT)
manifest=ROOT/'data'/'speaker_recognition'/'metadata.csv'
rows=[]
with manifest.open(encoding='utf-8-sig',newline='') as f:
 for row in csv.DictReader(f):
  if row['split']!='test': continue
  audio=load_wav(ROOT/'data'/'speaker_recognition'/row['file_name'],target_sample_rate=16000)
  features,_=prepare_utterance(audio.samples,audio.sample_rate)
  scores={sid:min(dtw_distance(features,t) for t in items) for sid,items in templates.items()}
  ordered=sorted(scores.items(),key=lambda item:item[1])
  best_id,distance=ordered[0]
  result={'speaker_id':best_id,'speaker_name':names.get(best_id,best_id),'distance':distance,'scores':{sid:{'name':names.get(sid,sid),'distance':value} for sid,value in ordered}}
  rows.append({'file_name':row['file_name'],'true_speaker_id':row['speaker_id'],'true_name':row['speaker_name'],'predicted_speaker_id':result['speaker_id'],'predicted_name':result['speaker_name'],'correct':result['speaker_id']==row['speaker_id'],'best_distance':result['distance'],'scores_json':json.dumps(result['scores'],ensure_ascii=False)})
out=ROOT/'outputs'/'experiment4_speaker';out.mkdir(parents=True,exist_ok=True)
with (out/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
ids=sorted(names)
cm={a:{b:sum(r['true_speaker_id']==a and r['predicted_speaker_id']==b for r in rows) for b in ids} for a in ids}
correct=sum(r['correct'] for r in rows)
summary={'experiment':'MFCC + DTW speaker recognition','speakers':names,'training_recordings':sum(len(v) for v in templates.values()),'test_recordings':len(rows),'correct':correct,'accuracy':correct/len(rows) if rows else 0,'confusion_matrix':cm,'limitations':'Evaluation uses the supplied held-out recordings from the five enrolled members; live microphone performance can differ.'}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
with (out/'confusion_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['true\\pred']+ids);[w.writerow([sid]+[cm[sid][p] for p in ids]) for sid in ids]
print(json.dumps(summary,ensure_ascii=False,indent=2))

