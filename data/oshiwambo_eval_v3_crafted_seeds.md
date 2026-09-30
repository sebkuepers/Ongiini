# v3 crafted subset — 100 phenomenon items for the 600-item v1.0 build

Adds 100 crafted items to the 110 in `oshiwambo_eval_v2_challenge_seeds.md`
so that every phenomenon tag reaches at least 30 items across the full
600-item set.

**Source style.** Plain English as a Namibian second-language speaker would
write it on WhatsApp: short sentences, common words, no native-speaker
idioms, no rare vocabulary. The difficulty must come from the target
language (noun classes, tense, register, pronouns), not from the English.
Formal items are the exception — they keep authentic Namibian officialese.

Tags are strict: an item carries a tag only where that phenomenon is the
real translation difficulty. Drafted with LLM assistance (disclose in the
paper); pending native-speaker review for naturalness.

Format per item: `EN | tags | word_count` — a domain tag (community,
religious) overrides the default `challenge` domain.

---

## idiom_nonliteral
Everyday figurative expressions that second-language English speakers in Namibia actually use (my phone is dead, money is tight, a big heart, the queue was a mile long) — not native-speaker idioms. The translator renders the meaning, not the words.

```
My phone is dead, so I will call you later. | idiom_nonliteral | 10
I am dying of hunger; what can I cook quickly? | idiom_nonliteral | 10
Money is tight this month, so we cannot buy new clothes. | idiom_nonliteral, negation | 11
Keep an eye on the pot while I go and fetch water. | idiom_nonliteral | 12
This heat is killing me; let us sit in the shade. | idiom_nonliteral | 11
We are all in the same boat, because the drought hit every farmer. | idiom_nonliteral | 13
I am broke until month-end, so I cannot come to town. | idiom_nonliteral, negation | 11
My head is spinning from all these forms. | idiom_nonliteral | 8
My heart is broken since my grandfather passed away. | idiom_nonliteral, community | 9
The exam paper was very long and it finished me. | idiom_nonliteral | 10
Stop going around in circles and tell me what happened. | idiom_nonliteral | 10
My grandmother has a heart of gold and feeds every child in the street. | idiom_nonliteral, community | 14
Losing that job opened my eyes, and now I run my own shop. | idiom_nonliteral | 13
I know someone at the office, so I will put in a good word for you. | idiom_nonliteral | 16
Our business is growing slowly, but we are on the right track. | idiom_nonliteral | 12
The queue at the clinic was a mile long this morning. | idiom_nonliteral | 11
My brother has a big heart and helps everyone in the village. | idiom_nonliteral, community | 12
```

## polysemy
The ambiguous word (charge, light, spring, match, fine, cell, play, fair, park, bill, board, well, break) decides the Oshiwambo lexeme; the rest of the sentence disambiguates.

```
Please charge my phone while I cook. | polysemy, code_switch | 7
The shop charges N$50 for delivery to the village. | polysemy, numbers_dates | 9
The police will charge him with theft tomorrow. | polysemy | 8
Switch on the light, it's getting dark. | polysemy | 7
This bag is light enough for a child to carry. | polysemy | 10
The spring near our cattle post has dried up. | polysemy, community | 9
Our team won the football match on Saturday. | polysemy | 8
Do you have a match to light the fire? | polysemy | 9
I had to pay a fine because my car licence had expired. | polysemy | 12
Please break the bread into small pieces. | polysemy | 7
My cell has no signal out here at the farm. | polysemy | 10
The prisoner spent the night alone in a cell. | polysemy | 9
The learners will perform a play about the history of our town. | polysemy | 12
It's not fair that only some learners received books. | polysemy, negation | 9
We will talk about it during the break. | polysemy | 8
You can park the car under the tree. | polysemy | 8
The electricity bill is higher this month. | polysemy | 7
The school board will announce the new principal next week. | polysemy | 10
Fetch water from the well before it gets dark. | polysemy | 9
She sings well, but she has never sung in church. | polysemy, negation, religious | 10
```

## politeness_register
Short forms of address: elder (Tate / Meme / Kuku), peer, child, office holder.

```
Good morning, Meme, how are you? | politeness_register | 6
Sorry to bother you, madam. | politeness_register | 5
Please greet your parents for me. | politeness_register, community | 6
Thank you, my child. | politeness_register | 4
Welcome, Tate, please sit down. | politeness_register | 5
Excuse me, Pastor, may I speak? | politeness_register, religious | 6
Children, greet the visitors properly. | politeness_register | 5
Kuku, did you sleep well? | politeness_register, community | 5
Forgive me, I didn't mean it. | politeness_register, negation | 6
Please, Meme, let me explain. | politeness_register | 5
Uncle, may we borrow your wheelbarrow? | politeness_register, community | 6
Bro, are you coming or not? | politeness_register | 6
Honourable Councillor, thank you for coming. | politeness_register | 6
```

## noun_class_agreement
Adjective, numeral and demonstrative concord across Bantu noun classes.

```
Those tall girls play netball. | noun_class_agreement | 5
The old woman sells sweet bread. | noun_class_agreement | 6
Those two old chairs are broken. | noun_class_agreement, numbers_dates | 6
These red shoes are too small. | noun_class_agreement | 6
That tall tree gives good shade. | noun_class_agreement | 6
The long river has many fish. | noun_class_agreement | 6
My two sisters bought five chickens. | noun_class_agreement, numbers_dates | 6
A small child lost one shoe. | noun_class_agreement | 6
Her beautiful dress has three buttons. | noun_class_agreement, numbers_dates | 6
Those big dogs chase our goats. | noun_class_agreement | 6
Our new house has three rooms. | noun_class_agreement, numbers_dates | 6
All the old pots are dirty. | noun_class_agreement | 6
The heavy rains destroyed our mahangu fields. | noun_class_agreement, community | 7
Big trucks and small cars use this road. | noun_class_agreement | 8
The young men fixed our broken roof. | noun_class_agreement | 7
The clinic's nurses helped many sick people. | noun_class_agreement | 7
Many young people have left the village. | noun_class_agreement, community | 7
These sweet oranges come from Rundu. | noun_class_agreement, named_entities | 6
These new phones are expensive. | noun_class_agreement | 5
The big white car belongs to our teacher. | noun_class_agreement | 8
Our small dog has four puppies. | noun_class_agreement, numbers_dates | 6
```

## pronoun_coreference
The antecedent must be resolved to choose the Oshiwambo pronoun, including across a sentence boundary.

```
Ask her if she saw it. | pronoun_coreference | 6
Give it to him, not her. | pronoun_coreference, negation | 6
Tell them we'll wait for them. | pronoun_coreference | 6
The farmer sold the cow after it got sick. | pronoun_coreference | 9
The car hit the wall, but it wasn't damaged. | pronoun_coreference, negation | 9
I put the milk in the fridge because it was warm. | pronoun_coreference | 11
The nurse phoned the mother because she was worried. | pronoun_coreference | 9
Tangeni told his brother that he had failed. | pronoun_coreference | 8
I lent him my book and he lost it. | pronoun_coreference | 9
He thinks she likes him, but she doesn't. | pronoun_coreference, negation | 8
My aunt gave my mother her old phone. | pronoun_coreference, community | 8
My sister called Meme last night. She was crying. | pronoun_coreference, multi_sentence, community | 9
The learners met the new teachers. They were nervous. | pronoun_coreference, multi_sentence | 9
Tell your brother the goats are back. He was worried. | pronoun_coreference, multi_sentence | 10
The doctor spoke to my father. He said it was serious. | pronoun_coreference, multi_sentence | 11
```

## code_switch
English loanwords that stay English in everyday Oshiwambo (WhatsApp, airtime, data bundle, combi, Wi-Fi). Same definition as v0.1 and the translator notes — no Oshiwambo words inside the English source.

```
The rain is very heavy today, so the combi will be late. | code_switch | 12
Send it to my WhatsApp. | code_switch | 5
Buy me airtime, please. | code_switch | 4
My data bundle is finished. | code_switch | 5
Is the combi to Ondangwa full? | code_switch, named_entities | 6
Top up my prepaid electricity, please. | code_switch | 6
The Wi-Fi password isn't working. | code_switch, negation | 5
```

## tense_aspect
Recent past, perfect continuous, habitual past, past progressive, pluperfect.

```
She was so happy when she heard she had passed matric. | tense_aspect | 11
She has just left for work. | tense_aspect | 6
It has been raining since morning. | tense_aspect | 6
I used to herd cattle. | tense_aspect | 5
We were sleeping when it started. | tense_aspect | 6
I had already eaten. | tense_aspect | 4
```

## multi_sentence
Short two-sentence messages (longer multi-sentence items come from v0.1, the mined and the formal subsets).

```
It's late. Let's go home. | multi_sentence | 5
```
