# ASL Gloss-to-Text Translator

## User Manual

**Authors:**  
John Reyvi Gravidez  
Stingray Songgay  
Hubert Uybaan  

---

## Table of Contents
1. Overview  
2. System Requirements  
3. How to Use the App  
4. How to Translate Gloss to Text  
5. Output Sections  
6. Important Notes  
7. What is ASL Gloss  
8. ASL Glossing Rules and Examples  

---

## 1. Overview

The **ASL Gloss-to-Text Translator** is a web-based tool developed as part of a thesis study that addresses the challenge of inaccurate contextual translations of American Sign Language (ASL) glosses into natural English text.  
Inaccurate translations hinder effective communication, particularly for the Deaf and signing communities who rely on gloss-based representations in various digital or academic settings.

This project explores the use of a **T5 Transformer model** to improve the contextual accuracy of gloss translations.  
The system was trained and evaluated using a pipeline that involved data generation, web scraping, and expert validation to ensure high-quality, linguistically sound inputs.

---


## 2. How to Use the App

### Step 1: Download the App  
Download the zip file from this link: [Download Translator](https://tinyurl.com/2s3sym68)

### Step 2: Unzip the App Folder  
Unzip `T5_Translator.zip` to any location (desktop, USB, etc).

### Step 3: Run the Translator  
- Open the folder  
- Double-click `run.bat`  

You should see a black terminal window open. The model may take **40–60 seconds** to load.  

---


## 3. What to Expect After Running

During loading, you may see warnings like:

```
The `lang_code_to_id` attribute is deprecated.  
The `fairseq_tokens_to_ids` attribute is deprecated.  
Special tokens have been added in the vocabulary.  
```

✅ These can be ignored — they don’t affect functionality.  

Once loaded:  
- Local app will run at: `http://127.0.0.1:7860`  
- To share publicly, set `share=True` in `launch()`  

---

## 4. How to Translate Gloss to Text

1. Open your browser and go to `http://127.0.0.1:7860`  
2. Enter gloss input in the text box (e.g., `HELLO. HOW ARE YOU-Q`)  
3. Click **Translate**  

### Output Sections
- **Translated Text** – Best generated sentence  
- **Alternative Translations** – Beam search alternatives  
- **Tokenization** – Input/output token structure  
- **Predicted Words** – Top predictions with confidence  

---

## 5. Important Notes

- App runs **entirely offline**  
- **40–60 second pause** is normal at startup  
- Deprecation warnings can be ignored  
- Closing the terminal will stop the app — keep it open  

---

## 6. Understanding ASL Gloss

ASL gloss is a way of writing down American Sign Language using simplified English words to represent each sign.  
It follows ASL grammar (not English) and excludes articles, auxiliary verbs, and sometimes pronouns.  

- **All caps** words represent signs  
- **ASL word order**: TIME → TOPIC → COMMENT  
- **WH-questions**: WH-word at the end  
- **Yes/No questions**: Add `-Q` at the end  

---

## 7. Glossing Rules

1. **Use ALL CAPS**  
   - ✅ `ME GO STORE`  
   - ❌ `Me go store`  

2. **Follow ASL Word Order**  
   - English: `I went to the store yesterday`  
   - ASL Gloss: `YESTERDAY ME GO STORE`  

3. **WH-Questions at the End**  
   - English: `Where is your bag?`  
   - ASL Gloss: `YOUR BAG WHERE`  

4. **Use Only Content Words**  
   - English: `I don’t like apples` → Gloss: `ME NOT LIKE APPLE`  

5. **Yes/No Questions: Use -Q**  
   - English: `Do you like pizza?` → Gloss: `YOU LIKE PIZZA -Q`  

---

## 8. ASL Gloss → Text Examples

| ASL Gloss                      | English Translation                     |
|--------------------------------|-----------------------------------------|
| YESTERDAY ME EAT APPLE         | I ate an apple yesterday.               |
| YOU GO SCHOOL TOMORROW -Q      | Are you going to school tomorrow?       |
| FATHER COOK DINNER             | My father is cooking dinner.            |
| YOU LIKE DOG -Q                | Do you like dogs?                       |
| MOTHER LOVE CHILD              | The mother loves the child.             |
| NIGHT ME GO DANCE              | I'm going to dance tonight.             |
| NOW WEATHER RAIN               | It is raining now.                      |
| ME SISTER FINISH STUDY         | My sister finishes her studies.         |
| YESTERDAY FRIEND NOT COME      | My friend didn’t come yesterday.        |
| YOU BUY BOOK WHERE             | Where did you buy the book?             |
| TOMORROW ME CLEAN ROOM         | I will clean the room tomorrow.         |
| YOUR PHONE WHERE               | Where is your phone?                    |
| HE GO SCHOOL EVERYDAY          | He goes to school every day.            |
| MOTHER GO STORE MOTHER BUY MILK| My mother went to the store to buy milk.|

---

## Authors

- John Reyvi Gravidez  
- Stingray Songgay  
- Hubert Uybaán  
