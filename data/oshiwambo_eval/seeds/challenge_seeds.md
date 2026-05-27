# v2 challenge subset — crafted phenomenon-coverage items

110 EN items deliberately probing 11 linguistic phenomena where machine
translation models systematically fail. Each item is tagged with one or
more phenomena it stresses, plus a word count. Length-biased toward
medium-long (the target eval-set sweet-spot 7–25 words).

The `build_eval_v2.py` script reads this file, parses the tag annotations,
and merges these items into the final v2 TSV. If we need to trim to fit
budget, drop items marked `[OPTIONAL]` first.

Format per item: `EN | tags | word_count`

---

## negation
Single negation, double negation, scope ambiguity. Negation flips
meaning silently and is the #1 MT failure mode.

```
I didn't apply for the scholarship this year. | negation | 8
She said she wouldn't come to the meeting. | negation, pronoun_coreference | 9
We don't have any forms left at this office. | negation | 9
Nobody told me the deadline had moved. | negation, tense_aspect | 8
Don't forget to bring your ID, otherwise they won't help you. | negation, code_switch | 11
I never received the confirmation message you mentioned. | negation, tense_aspect | 9
It's not that I don't want to help — I just can't right now. | negation | 14
If you don't pay by Friday, the application will not be processed. | negation, numbers_dates | 12
Neither the principal nor the teacher could explain the new rules. | negation, named_entities | 11
He didn't say no, but he didn't say yes either. | negation | 11
```

## noun_class_agreement
Items with multiple nouns of different Bantu noun classes — verb/object/
adjective concord chains stress the model.

```
The teacher gave the children new books and pencils. | noun_class_agreement | 9
My mother carried the heavy bag from the market to our house. | noun_class_agreement, community | 13
The water in those bottles tasted strange to me yesterday. | noun_class_agreement, tense_aspect | 10
Those two old men sat under the tree near the river. | noun_class_agreement, numbers_dates | 11
The four young women learned how to repair the broken radio. | noun_class_agreement, numbers_dates, tense_aspect | 11
Three goats and five chickens were stolen from our farm last night. | noun_class_agreement, numbers_dates | 12
I sent the documents with my brother to the lawyer's office. | noun_class_agreement, formal | 11
These small children love the bright pictures in your storybook. | noun_class_agreement | 10
The school principal called all the parents to the hall on Monday. | noun_class_agreement, named_entities | 12
We need fresh vegetables, clean water, and warm clothes for the family. | noun_class_agreement, community | 12
```

## pronoun_coreference
Ambiguous pronouns where context (or noun class) disambiguates. Bantu
noun-class pronouns force the translator to pick a referent.

```
When my sister visited her, she gave her a beautiful necklace. | pronoun_coreference | 11
The teacher told the student that she had made an error. | pronoun_coreference, tense_aspect | 11
He picked up his phone and called his brother because he was worried. | pronoun_coreference, code_switch | 13
Mother said her colleague would come, but she didn't tell me when. | pronoun_coreference, negation | 12
If the patient calls the nurse, ask her to wait by the door. | pronoun_coreference | 13
She took her daughter to the doctor because she had a fever. | pronoun_coreference | 12
The principal met the parent, and he agreed to the new schedule. | pronoun_coreference | 12
We saw the cars before they left the parking lot. | pronoun_coreference, code_switch | 10
After he saw the dog, the man called his friend to help him. | pronoun_coreference | 13
My friend gave my brother her notebook because he had lost his. | pronoun_coreference, tense_aspect | 12
```

## tense_aspect
Perfect, recent past, habitual, future-perfect — Bantu often marks these
with distinct verb-stem extensions that English collapses.

```
I have been waiting for this letter for two weeks. | tense_aspect, numbers_dates | 10
She used to walk to school every day before the rains came. | tense_aspect | 12
We always pray before eating dinner together. | tense_aspect, religious | 7
By the time you arrive, I will have finished cleaning the kitchen. | tense_aspect | 12
He had already left when the principal called for him. | tense_aspect, pronoun_coreference | 10
I was about to send the email when the power went out. | tense_aspect, code_switch | 12
She has been studying nursing for three years now. | tense_aspect, numbers_dates | 9
Every Sunday after church we visited our grandmother. | tense_aspect, religious, community | 8
I just received your message — sorry for the late reply. | tense_aspect | 10
By next month I will have lived in Windhoek for five years. | tense_aspect, numbers_dates, named_entities | 12
```

## numbers_dates
Currency, dates, times, phone numbers, IDs — domain-formal items the
model must render precisely.

```
The application closes on the 15th of October at 5pm. | numbers_dates, formal | 10
I have saved N$2,750 for the deposit on my new apartment. | numbers_dates | 11
The meeting is scheduled for Tuesday at 9:30 in the morning. | numbers_dates, formal | 11
We need to vaccinate 1,200 children before the end of the month. | numbers_dates, formal | 12
My son was born on 23 April 2018, weighing 3.4 kilograms. | numbers_dates, community | 11
The hospital bill came to N$487 and we cannot afford it. | numbers_dates, negation | 11
She lost her ID number 78090604120 last week at the market. | numbers_dates | 11
We have only 14 days left to submit the form to BIPA. | numbers_dates, named_entities | 12
I waited two hours and forty-five minutes before being called. | numbers_dates | 10
The first instalment of N$1,500 is due on the 1st of June. | numbers_dates, formal | 12
```

## named_entities
Namibian places, ministries, common Namibian names. The translator
keeps these intact and adjusts surrounding grammar.

```
The clinic in Oshakati only opens from Monday to Friday. | named_entities, numbers_dates | 10
My brother works for the Ministry of Health and Social Services in Windhoek. | named_entities, formal | 12
We took the bus from Ondangwa to Outapi for the wedding. | named_entities, community | 11
Tate Shikongo asked me to deliver this message to you. | named_entities, politeness_register | 10
The new bridge near Rundu is finally finished. | named_entities | 8
She studies education at the University of Namibia main campus. | named_entities, formal | 10
BIPA refused my application because of a missing signature. | named_entities, formal | 9
I grew up in Eenhana but moved to Walvis Bay for work. | named_entities, tense_aspect | 12
Meme Ndapewa is the best baker in our village. | named_entities, politeness_register, community | 10
The road from Katima Mulilo to Tsumkwe is still very bad. | named_entities | 11
```

## code_switch
English loanwords that have entered everyday Oshindonga — phone/tech
vocabulary, government/banking terms, social-media nouns.

```
Please send me the link via WhatsApp when you have a moment. | code_switch | 12
I forgot my password and cannot access my email account. | code_switch, negation | 10
Can you upload the PDF to the school portal before midnight? | code_switch, numbers_dates | 11
The grant from the Ministry is paid into your bank account every month. | code_switch, formal | 13
My boss approved my leave for the funeral next week. | code_switch | 10
The internet connection at the library is faster than at home. | code_switch | 11
Please save the document as a backup before you make changes. | code_switch | 11
I need to renew my driver's licence at NaTIS this week. | code_switch, named_entities | 11
The principal sent the timetable in a Google Doc yesterday. | code_switch, named_entities | 10
Use the QR code on the receipt to access the warranty information. | code_switch, formal | 12
```

## politeness_register
Elder/peer/child address, formal/informal — Oshiwambo has strong
honorific norms (Tate/Meme/Kuku/Tatekulu) the translator must pick.

```
Tate, can you please explain how this medicine should be taken? | politeness_register | 11
Meme, I hope you are well; my mother sends her warmest greetings. | politeness_register, community | 12
Hey kid, come here quickly and show me what you've drawn! | politeness_register | 11
Sir, I would like to request a meeting with you at your convenience. | politeness_register, formal | 13
My friend, you really helped me out today — I owe you one. | politeness_register, idiom_nonliteral | 13
Honourable Minister, please consider our village's request for clean water. | politeness_register, formal | 11
Babe, what time should I pick you up tonight? | politeness_register | 10
Dear customer, your appointment has been rescheduled to next Tuesday. | politeness_register, formal, numbers_dates | 10
Hi guys, just confirming we're still on for Saturday. | politeness_register, numbers_dates | 9
Kuku, please tell us another story about when you were young. | politeness_register, community, tense_aspect | 11
```

## idiom_nonliteral
English idioms — the translator either finds an Oshindonga idiom that
maps, or paraphrases. Tests cultural-linguistic competence.

```
Break a leg at your job interview tomorrow! | idiom_nonliteral | 8
It costs an arm and a leg to live in Windhoek these days. | idiom_nonliteral, named_entities | 13
Don't put all your eggs in one basket — apply to multiple universities. | idiom_nonliteral, negation, formal | 12
She has a green thumb; everything she plants grows. | idiom_nonliteral | 9
He's barking up the wrong tree if he thinks I will help him. | idiom_nonliteral, pronoun_coreference | 13
Let's call it a day — we have worked long enough. | idiom_nonliteral, tense_aspect | 10
Better late than never — at least you came to the meeting. | idiom_nonliteral | 11
Don't burn your bridges with that company; you may need them later. | idiom_nonliteral, negation | 12
She let the cat out of the bag about the surprise party. | idiom_nonliteral | 12
Time flies when you are having fun with friends. | idiom_nonliteral | 9
```

## polysemy
Same EN word, different meaning per context. Translator must read the
sentence carefully and pick the correct Oshindonga lexeme.

```
Please book a room for two nights at the hotel near the beach. | polysemy, numbers_dates | 13
I need to book a doctor's appointment for my mother next week. | polysemy, community, numbers_dates | 12
The bank is closed today because of the public holiday. | polysemy, formal | 10
We sat on the river bank watching the sunset together. | polysemy, community | 10
She turned right at the school and continued for two kilometres. | polysemy, numbers_dates | 11
It is your right to receive proper medical care at the clinic. | polysemy, formal | 12
I work at a school as a teacher of the lower grades. | polysemy | 12
I think the old school of teaching was actually quite effective. | polysemy, tense_aspect | 11
Please change the date on the application form to next Monday. | polysemy, formal, numbers_dates | 11
Can you give me change for fifty dollars? | polysemy, numbers_dates | 8
```

## multi_sentence
2–4 sentence mini-paragraphs testing discourse cohesion: pronoun
chains, temporal sequencing, topic continuity.

```
I went to the hospital yesterday. They told me to come back today. I really hope I won't have to wait as long this time. | multi_sentence, tense_aspect, negation | 26
The school called my husband while he was at work. They said our daughter was not feeling well. He went to fetch her immediately. | multi_sentence, pronoun_coreference, community | 22
I sent in my application last month. Since then I have heard nothing back. Should I follow up with them or just wait? | multi_sentence, formal, negation | 23
Thank you for your message. I will check with my supervisor tomorrow morning. I will let you know what she says by the end of the day. | multi_sentence, formal, pronoun_coreference | 27
We are organising a small celebration for my mother's 60th birthday. It will be at our home on Saturday at 2pm. Please come if you can — she would love to see you. | multi_sentence, community, numbers_dates | 33
The bus broke down halfway. We had to walk for almost two hours in the heat. By the time we arrived, the meeting was already over. | multi_sentence, tense_aspect, numbers_dates | 25
My CV is attached to this message. I have included references from my previous two employers. Please let me know if you need anything else. | multi_sentence, formal, code_switch | 25
Don't forget your raincoat. The weather forecast says it will rain heavily this afternoon. Better safe than sorry. | multi_sentence, negation, idiom_nonliteral | 18
I tried calling you three times. Maybe your phone is off, or maybe you are in a meeting. Please call me back when you can. | multi_sentence, code_switch, numbers_dates | 24
She didn't come to school yesterday. Today she still isn't here. I am starting to get worried about her. | multi_sentence, negation, tense_aspect | 19
```
