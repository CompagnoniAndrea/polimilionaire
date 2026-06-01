Il codice è composto dalla cartella millionaire_client che non va toccata e fornisce le api per connettersi al sito. 
L'accesso e l'avvio della partita avvengono tramite la classe  `MillionaireClient`. Legge automaticamente le credenziali da un file `.env` e gestisce l'accesso, da questo punto in poi ho l'oggetto `game` che ha tutto ciò che serve per giocare 
from millionaire_client import MillionaireClient
```python
# 1. Creazione del "collegamento" verso l'indirizzo del server
client = MillionaireClient("http://131.175.15.22:51111/")

# 2. Login
user = client.login(username, pwd)

# 3. Avvio di una nuova partita per la competizione numero 1
game = client.game.start(competition_id=1)
```

Come importare un modello, vediamo l'esempio: 
```python
from transformers import pipeline
import torch

llm_pipeline = pipeline(
    "text-generation",              # Task: genera testo successivo
    model="Qwen/Qwen3-2B-Instruct", # Nome del LLM open source
    torch_dtype=torch.bfloat16,     # Per usare meno VRAM sulla scheda video
    device_map="auto"               # Sposta in automatico su GPU se disponibile
)
```
Voglio portare avanti questo filo narrativo mostrando, passo dopo passo, gli esperimenti, i loro risultati, i fallimenti (se ci sono), un paragrafo con le nostre considerazioni e infine il passo successivo con i milgioramenti. Questo per me definisce una pipeline concreta in cui si possono vedere tutti i varti step che abbiamo compiuto e i ragionamenti/osservazioni che abbiamo eseguito. 
Insomma prima stabiliamo che il problema esiste (baseline), poi capiamo dove il modello sbeglia. Poi si capisce perché sbaglia, si provano vari strumenti per milgiorare le performance e infine si combina tutto in una stretgy finale e si verifica che il sistema complessivo è meglio di quello delle piccole parti. Ogni sezione deve rispondere a una domanda precisa: 

**Per ogni esperimento fare più di una run per modello per vedere come il modello si comporta per diversi epserimenti, uno non è significativo**

- Esperimento1 ->
	- Costruisco la Baseline (provo un modello e plotto i risultati,  l'obiettivo è capire quanto è bravo il modello così com'è)

- Esperimento2 → 
	- Confronto modelli + thinking vs non-thinking. Si può anche vedere il thinking su quali categorie performa meglio e su quali no e adottare una strategia dinamica (Devo capire quale modello scegliere come baseline, essendocene moltissimi voglio capire quale si comporta bene. Anhce qua plotto le statistiche ottenute per ogni modello e arrivo a delle conclusioni.) Forse è meglio per ogni modello fare 3 run di gioco completo e calcolare media e deviazione standard in modo da avere risultati che sono più statisticamente sensati 
	
- Esperimento3 → 
	- Prompt engineering (heatmap modello × prompt), l'obiettivo è capire se il prompt cambia qualcosa e se si qual è il prompt migliore. Inoltre capiamo anche che prompt migliore in base a zero shot, one-shot, few-shot...

- Esperimento4 ->
    - Quantizzazione, se riduco la qualità con cui rappresento i pesi e le attivazioni (da floating point 32 bit a 16 bit oppure 8 bit), cambia di tanto l'accuracy? me lo posso permettere per milgiorare di molto il tempo e usare modelli con molti più parametri?

- Esperimento5  → 
	- Analisi categorie approfondita (matematica in particolare), l'obiettivo è capire dove il modello sbaglia di più e capire il perché

- Esperimento6  → 
	- Analisi per difficoltà + overconfidence, il modello sbaglia di più quando è difficile? Il modello con quanta confidenza risponde quando sa e quando non sa, calcolo la confidenza vs accuracy reale 

- Esperimento7  →
	- RAG con Wikipedia o altre fonti, vediamo le statistiche tra RAG e senza paragonate, capiamo quando lo usa e quando no, quando peggiora o migliora 
	
- Esperimento8  → 
	- Esperimento chiamando diversi tool con langchain 
	
- Esperimento9  →
	- Fine-tuning LoRA prendiamo un dataset significativa e finetuniniamo il modello su quello e vediamo se ci sono dei miglioramenti
	
- Esperimento10 → 
    - Confronto finale, tabella riassuntiva (con accuracy, note...), cosa ha funzionato, cosa no, perché e come ci siamo mossi a riguardo. 

--- 

Butto giù una stretgia per loggare tutti i risultati e poi plottarli, in questo modo abbiamo una pipeline di generazione che ci permette di capire quando e come abbiamo fatto gli esperimenti e prendere il loro percorso, inserirlo nella funzioni di plotting e avere i risultati 
 
Questi sono i dati che ho intenzione di loggare in modo da avere una panoramica d'insieme:
 
| Campo | Esempio | Perché |
|---|---|---|
| `run_id` | `run_20260521_1430_baseline_mistral7b` | Identificatore unico della sessione |
| `timestamp` | `2026-05-21T14:30:22` | Quando è stata fatta la run |
| `experiment_type` | `baseline`, `rag`, `tool`, `finetune` | Per filtrare per tipo di esperimento |
| `model_name` | `mistral-7b-instruct` | Quale modello |
| `model_size` | `7B` | Per analisi dimensione vs accuracy |
| `quantization` | `4bit`, `8bit`, `none` | Impatta la qualità del modello |
| `prompt_type` | `zero_shot`, `few_shot`, `cot` | Quale prompt è stato usato |
| `rag_used` | `true` / `false` | Per confronto RAG vs no RAG |
| `tool_used` | `true` / `false` | Per confronto agente vs baseline |
| `fine_tuned` | `true` / `false` | Modello fine-tunato o no |
| `notes` | `"Prima run, T4 GPU"` | Note libere sulla sessione |
 
Questi sono i dati che mi salvo per ogni domanda in modo da ricostruire l'output del modello:

| Campo | Esempio | Perché |
|---|---|---|
| `question_id` | `1` → `15` | Posizione nella sessione |
| `category` | `matematica` | Per analisi per categoria |
| `question_text` | `"Quanto fa 17% di 340?"` | Per debug e analisi qualitativa |
| `options` | `A:51.2, B:57.8, ...` | Le 4 risposte disponibili |
| `correct_answer` | `B` | Ground truth |
| `model_answer` | `A` | Risposta del modello |
| `is_correct` | `true` / `false` | Derivato, ma utile averlo esplicito |
| `confidence_of_each_answer` | `"A": 0.74, "B": 0.18,"C": 0.05,"D": 0.03`| Probabilità di ciascuna risposta, per vedere quanto ero lontano da quella corretta|
| `response_time_sec` | `4.2` | Tempo per la risposta |
| `rag_context` | `"Parigi è la capitale..."` | Contesto recuperato (se RAG) |
| `tool_result` | `57.8` | Output della calcolatrice (se tool) |
 
 
La struttura che ho pensato è questa: ogni run genera un file json che salvo nella cartella runs. Quando ho bisogno di plottare dei grafici definisco all'inizio delle funzione di plotting e poi basta che le chiamo con i relativi path del json che mi interessano.
Il notebook è diviso per sezioni, onuna con il suo relativo esperimento. 

```
polimillionaire/
│
├── notebook.ipynb
│
├── runs/                        ← un JSON per run di gioco
│   ├── run_20260521_1430_baseline_mistral7b.json
│   ├── run_20260522_0915_prompt_mistral7b_cot.json
│   ├── run_20260523_1100_rag_mistral7b.json
│   └── ...
```
Ogni file JSON ha questa struttura che avevo già anticipato prima: 
```json
{
  "run_metadata": {
    "run_id":          "run_20260521_1430_baseline_mistral7b",
    "timestamp":       "2026-05-21T14:30:22",
    "experiment_type": "baseline",
    "model_name":      "mistral7b",
    "model_size":      "7B",
    "quantization":    "4bit",
    "prompt_type":     "zero_shot",
    "rag_used":        false,
    "tool_used":       false,
    "fine_tuned":      false,
    "notes":           "Prima run baseline"
  },
  "session_stats": {
    "total_questions":       8,
    "correct":               5,
    "accuracy":              0.625,
    "mean_confidence":       0.81,
    "mean_response_time_sec": 3.4,
    "max_level_reached":     8
  },
  "questions": [
    {
      "question_id":       1,
      "level":             2,
      "category":          "matematica",
      "question_text":     "Quanto fa 17% di 340?",
      "options": {
        "A": "51.2",
        "B": "57.8",
        "C": "62.0",
        "D": "47.6"
      },
      "correct_answer":    "B",
      "model_answer":      "A",
      "is_correct":        false,
      "answer_probs": {
        "A": 0.74,
        "B": 0.18,
        "C": 0.05,
        "D": 0.03
      },
      "confidence":        0.74,
      "response_time_sec": 4.2,
      "rag_context":       null,
      "tool_result":       null
    }
  ]
}
```

Per ogni run avrò un codice di logging che automaticamente inserisce tutte queste informazioni.
Come grafici avevo pensato a questi: 

- accuracy per modello con deviazione standard (almeno due json di due run diverse in input)
- accuracy per categoria (gli basta un json di una run in input)
- heatmap_model: heatmap che mostra per ogni prompt e modello l'heatmap dell'accuracy (CONFUSION MATRIX con le lettere )
- confidence distribution: quanto è confidence a rispondere a errate e giuste 
- quanto era vicino a farla giusta il modello per categoria
- rag comparsion per categoria 


I nomi delle run saranno fatti così: 
```
run_{YYYYMMDD}_{HHMM}_{experiment_type}_{model_name}.json
```

Queste sono alcune domande con cui può essere strurrato il notebbok e che possono aiutare a rispodnere e a trovare una narrativa da portare, in mopdo che sia chiara la pipeline che stiamo seguendo
1) Bigger is always better?
2) la quantizzazione vale il trade of con le performance?
3) I modelli "thinking" (DeepSeek-R1, QwQ) ragionano meglio ma ci stanno nel timeout di 30 secondi e sono davvero più utiili (e se si su tutte lòe task o no)?
4) se cambio solo il tono del prompt (formale vs casual) l'accuracy cambia? 
5) Few-shot aiuta di più su domande difficili o su quelle facili?
6) Il modello sbaglia i calcoli matematici perché non sa fare i conti, o perché non capisce la domanda?
7) Gli errori (matematici e NON) sono sistematici (sempre stesso tipo di sbaglio) o casuali?
8) Il RAG migliora l'accuracy solo quando trova contesto davvero rilevante, o anche un contesto vago aiuta?
9) Quando il RAG peggiora le performance, il modello si è distratto dal contesto sbagliato dato dal RAG o aveva già la risposta giusta senza contesto?
10) Conta più la qualità del contesto recuperato o la lunghezza?
11) Il modello è overconfident in modo uniforme su tutte le categorie, o è molto confident su alcune e pessimo su altre?
12) Quando il modello sbaglia, la risposta corretta era almeno la sua seconda scelta oppure sbagliava di moltissimo?


Altre immagini che possono aiutarmi nell'analisi: 
- Esempio di prompt mostrato visivamente il testo del prompt zero-shot vs CoT fianco a fianco, con la risposta del modello sotto. Fa capire subito perché uno funziona meglio dell'altro
- Esempio di RAG fallito, la domanda, il contesto recuperato da Wikipedia che era irrilevante, e la risposta sbagliata che ne è derivata. Vale più di mille parole
- Tabella degli errori più frequenti, le 5-10 domande che il modello ha sbagliato con più confidenza, con domanda, risposta data, risposta corretta e confidenza. Immediato da leggere

Queste sono tutte cose a cui sto pensando ora prima di inizare con la vera e propria implemnetazione, potrebbero non avere senso o essere sbagliate. Lo scopo di ciò è buttare giù un piano di azione per costruire un'architettura/pipeline sensata