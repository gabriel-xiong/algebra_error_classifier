# Algebra Error-Type Classifier BrainLift

## Owners
Gabriel Xiong

## Purpose
To ground the design of a fine-tuned small-model error-type classifier for K-12 algebra in the learning-science literature, and to establish the behavior thesis for the project: that reliable error typing comes from a coarse, well-grounded taxonomy and calibrated abstention, not from model scale. The target behavior is that, given an algebra problem, the correct answer, and a student's incorrect answer (ideally with shown work), the model returns one error-type label from a fixed taxonomy with a confidence score, and abstains when the observable signal does not support a confident label.

### In Scope
- Which established error-analysis frameworks exist, and which ones fit symbolic equation-solving versus word problems
- The recurring algebra-specific error families and where they come from in the research
- Why some error types are not distinguishable from the observable signal a deployed system actually has
- How the granularity of a taxonomy affects labeling reliability
- What is recoverable from final-answer-only signal versus shown-work signal
- The candidate label set for the classifier and how to validate it

### Out of Scope
- The MCAT and Speedrun product context (covered in a separate BrainLift)
- QLoRA and training-loop mechanics, hyperparameter tuning (the assignment treats training as a downstream button-press)
- Teaching or tutoring behavior (this is a classifier, not a tutor)
- Broad algebra beyond one-variable linear equations (the domain is kept narrow by design)

---

## DOK 4: Spiky Points of View (SPOVs)

**Spiky POV 1: The hard part of error classification is the taxonomy, not the model. An ungrounded label set produces inconsistent labels regardless of model size.**

A wrong answer to 3x + 5 = 9 could come from a sign error, a wrong inverse operation, or a one-off arithmetic slip, and each carries completely different pedagogical meaning. The instinct in a one-week ML build is to invent five or six error categories and point a model at them. That is backwards. When the label ontology is ungrounded, categories overlap, the true label is under-determined by what is observable, and the same input receives different labels across runs no matter how capable the model is. (This pairs two results rather than one paper: Warrens (2010) shows that overlapping or finer categories lower inter-rater agreement, and Liu et al. (2023) shows a capable model does not rescue a poor ontology, with accuracy falling from 91.9% to 39.8% as the label set grows.) The algebra-misconception literature already did this work: Booth, Barbieri, Eyer and Pare-Blagoev (2014) coded the work of 565 Algebra I students into six conceptual error families that recur across four decades of independent research (Kieran on the equals sign, Kuchemann on variables, MacGregor and Stacey on conjoining). Those families are defined by what is visible in written work, which is exactly the property a classifier needs. The dataset is only as good as the taxonomy it encodes, so the taxonomy is the real deliverable, not the training run.

**Spiky POV 2: Granularity is a tradeoff, not a maximization. The right error taxonomy is deliberately coarse.**

More categories feel more precise, but finer distinctions are exactly where labels become under-determined and where both humans and models start disagreeing. Liu, Sonkar, Wang, Woodhead and Baraniuk (2023) found GPT-4's misconception-identification accuracy fell from 91.9% when choosing among 4 candidate misconceptions to 39.8% when choosing among 100, an inverse relationship between label count and accuracy. Warrens (2010) proved formally that merging categories generally raises inter-rater agreement. This cuts against the intuition that a richer taxonomy is a better one. For a small model working from a thin signal, a coarse set of six or seven labels is not a compromise, it is the design. (Honesty flag: the coarse-is-more-reliable claim is strongly supported in theory and in adjacent domains, but I did not find a study measuring it inside a mathematics-error taxonomy specifically, so this is a well-supported working assumption, not a settled result for this exact domain.)

**Spiky POV 3: The observable signal is thin, so error classification has a hard accuracy ceiling. The correct design target is calibrated abstention, not maximum accuracy.**

In a system that could actually ship, you get the problem, the wrong answer, and maybe the shown work. You never get the student's intention. The slip-versus-mistake distinction (Norman 1981, Reason 1990) is defined by intention: a careless slip and a stable misconception can produce the identical wrong answer. McNichols, Zhang and Lan (2023) state directly that without knowledge of student intent it is impossible to be certain about the true cause of an error, and their own classifier confuses two specific error types 41% of the time because the true intent is not recoverable from the trace. Adding shown-work signal raises the ceiling but does not remove it (their accuracy rose from 82.0% to 85.9% when student-action context was added). A classifier that always emits a confident label is therefore wrong a predictable fraction of the time. The right behavior is to label confidently where the signal supports it and abstain where it does not, and the evaluation has to reward calibration, not just raw accuracy.

---

## Experts

**Expert 1: Julie L. Booth**
- Who: Professor of Educational Psychology at Temple University.
- Focus: Algebra learning, the role of misconceptions in algebra performance, and worked-example and error-reflection interventions.
- Why Follow: Booth et al. (2014) is the direct anchor for the candidate taxonomy. Its six conceptual error families are the best-matched, peer-reviewed scheme for the middle-school-to-Algebra-I population this project targets, and Booth and Koedinger (2008) established that equals-sign and negative-sign misconceptions predict both performance and learning.
- Where: https://education.temple.edu/about/faculty-staff/julie-booth-tuh40746

**Expert 2: Dietmar Kuchemann**
- Who: Mathematics education researcher, associated with the CSMS (Concepts in Secondary Mathematics and Science) study in the UK.
- Focus: How students interpret algebraic letters and variables.
- Why Follow: Kuchemann (1981) defined the six interpretations of letters (evaluated, ignored, object, specific unknown, generalized number, variable) that underlie the "variable error" family in the taxonomy, including the letter-as-object thinking that drives conjoining and reversal errors.
- Where: CSMS work, in Hart (ed.), "Children's Understanding of Mathematics: 11-16" (1981).

**Expert 3: Carolyn Kieran**
- Who: Professor Emeritus, Departement de mathematiques, Universite du Quebec a Montreal.
- Focus: The learning and teaching of school algebra, including the transition from arithmetic to algebra.
- Why Follow: Kieran (1981) is the foundational source for the operational-versus-relational understanding of the equals sign, which grounds the "equality/balance error" category. Her work explains why students treat "=" as a "do something" operator rather than a statement that two sides are equal.
- Where: Kieran (1981), "Concepts associated with the equality symbol," Educational Studies in Mathematics 12(3).

**Expert 4: Donald A. Norman**
- Who: Cognitive scientist; foundational figure in human error research and design.
- Focus: The cognitive structure of human error.
- Why Follow: Norman (1981), "Categorization of Action Slips" (Psychological Review), established the slip-versus-mistake distinction that is the theoretical basis for SPOV 3. The definition of a slip as an execution error with correct intention is exactly why a final answer cannot reveal whether an error was careless or conceptual.
- Where: Norman (1981), Psychological Review 88(1). See also Reason (1990), "Human Error."

**Expert 5: Kurt VanLehn**
- Who: Professor of Computer Science, Arizona State University; a founder of the intelligent-tutoring-systems field.
- Focus: Systematic procedural bugs, repair theory, and student modeling.
- Why Follow: Brown and VanLehn (1980) repair theory documented that systematic bugs are real but unstable, and that the same underlying misconception surfaces as different observable bugs across problems ("bug migration"). This is the direct source for the caution that the map from observable error to latent cause is many-to-many, which supports the abstention design.
- Where: Brown and VanLehn (1980), "Repair Theory," Cognitive Science 4(4).

**Expert 6: Andrew S. Lan**
- Who: Assistant Professor, College of Information and Computer Sciences, University of Massachusetts Amherst.
- Focus: Machine learning for education, including automated error and misconception classification.
- Why Follow: McNichols, Zhang and Lan (2023) is the most direct prior work on this exact task (classifying algebra errors from student steps with language models). It provides both the realistic accuracy bar (mid-80s on multi-label schemes with shown work) and the clearest statement of the observability ceiling that motivates SPOV 3.
- Where: https://arxiv.org/abs/2305.06163

---

## DOK 3: Insights

### From Category 1: Error-Analysis Frameworks
**Insight 1:** Newman's Error Analysis is a process-stage model built for word problems, and it collapses on bare symbolic equations. When applied to solving something like 3x + 5 = 9, nearly every error lands in the single "process skills" stage, because there is almost no reading or comprehension load. A framework's fit is a function of the task, so importing a word-problem taxonomy onto equation-solving would yield one useful category and four dead ones.

**Insight 2:** The frameworks that survive contact with equation-solving are content-specific, not process-stage. Booth's families are defined by what is visible in the written work (a dropped negative, a conjoined term, an unbalanced operation), which is the exact property a classifier operating on shown work can key on. The right framework for this project describes what the error looks like on the page, not where in an interview it would surface.

### From Category 2: Algebra-Specific Misconceptions
**Insight 3:** The algebra-misconception literature converges. Independent researchers over roughly forty years keep rediscovering the same small set of error families: equals-sign, variable, negative-sign, distribution, and conjoining. Convergence across independent sources is the strongest available evidence that a taxonomy is carving the space at real joints rather than at convenient ones, and it is what makes SPOV 1's "grounded" claim more than an appeal to authority.

**Insight 4:** Most of these errors are rooted in stable conceptual misconceptions (letter-as-object, the operational equals sign), not carelessness. Because a stable misconception produces the same fingerprint across many different students and problems, these error types are in principle learnable from data. The systematic ones are the learnable ones, which is precisely why the taxonomy should separate them from the arithmetic slip that is not systematic.

### From Category 3: Slips, Mistakes, and the Observability Problem
**Insight 5:** The slip-versus-mistake distinction is defined by intention, and intention is not in the data. This is the hard wall behind SPOV 3: no amount of model capability recovers a signal that was never present in the input. A one-off slip and a stable misconception can be byte-for-byte identical on the page.

**Insight 6:** Systematic bugs are real but unstable. Bug migration (Brown and VanLehn) means one underlying misconception surfaces as different observable errors across problems, so the mapping from observable to cause is many-to-many. Any design that claims a clean one-to-one diagnosis from a single response is overstating what the data can support, which is a second, independent reason to build in abstention.

### From Category 4: Granularity and Reliability
**Insight 7:** Accuracy and taxonomy size trade off sharply. Liu et al.'s drop from 91.9% to 39.8% as candidates grew from 4 to 100 shows a finer taxonomy is not a free upgrade: it buys resolution at the direct cost of reliability, and past a point the labels stop meaning anything consistent. This is the quantitative backbone of SPOV 2.

**Insight 8:** Coarser schemes are structurally easier to agree on (Warrens proved merging categories generally raises agreement). This means reliability is partly a property of a taxonomy's shape, not only its grounding. A well-grounded taxonomy can still be made unreliable simply by splitting it too finely, so grounding and coarseness are two separate levers that both have to be pulled.

### From Category 5: Recoverability and Machine Classification
**Insight 9:** More observable signal raises the ceiling but does not remove it. McNichols et al. moved from 82.0% to 85.9% by adding action context, yet still confused one specific pair of error types 41% of the time. This quantifies SPOV 3: shown work helps, but a residual ambiguity is permanent, and that residual is exactly the region where abstention belongs rather than a forced guess.

**Insight 10:** The final answer alone is weakly diagnostic, because multiple distinct errors collapse to the same number. A specific wrong answer is only reliably diagnostic when the item was engineered for it, as in cognitive diagnostic models with tagged distractors, which a general free-response equation is not. So the project's confidence should scale with how much of the student's work is visible, not just with the model.

**Insight 11:** The prior ML work on this exact task already reports mid-80s accuracy on multi-label schemes with shown work. That sets a realistic bar and reframes the win. The point of a one-week fine-tune is not to beat that number, it is to show a clean base-versus-tuned delta on a coarse, grounded taxonomy with honest abstention behavior. Measuring against frontier capability would be measuring the wrong thing.

---

## DOK 2: Knowledge Tree

### Category 1: Error-Analysis Frameworks and Their Fit

#### Subcategory 1.1: Newman's Error Analysis (process-stage model)
**Source 1: Newman (1977, 1983) framework, as applied and summarized in subsequent analyses**

DOK 1 - Facts
- Newman's Error Analysis (NEA) locates a written-problem error at the first of five sequential hurdles a student fails: Reading, Comprehension, Transformation, Process Skills, and Encoding.
- Later formalizations (Clements; White) add Carelessness and Motivation as categories that sit outside the five-stage hierarchy because they can occur at any stage.
- When NEA is applied to symbolic or procedural mathematics rather than word problems, errors concentrate in the later stages. One linear-programming study reported the pattern reading 7.3%, comprehension 35.4%, transformation 47.9%, process skills 66.7%, encoding 85.4%.
- NEA relies on a structured interview protocol to assign categories reliably.

Summary and Analysis
NEA is a process-stage framework designed for word problems, where reading and comprehension carry real load. On bare one-variable equations, reading and comprehension errors are near zero and the mass of errors collapses into "process skills," so the framework offers little discrimination for this project's task. Its dependence on interviews is also disqualifying for a classifier that only sees written work: the signal NEA needs to assign a category is exactly the signal a deployed system does not have. NEA is useful as background on the process view and for two ideas worth borrowing as sub-tags (transformation and encoding), but it is not recommended as the primary label set.
Link to Source: https://files.eric.ed.gov/fulltext/EJ1059995.pdf

Reliability note (honesty flag): I did not find a published inter-rater reliability figure (for example a Cohen's kappa) validating Newman's five categories as a written-work-only coding scheme without interviews. A 2025 expert-weighted comparison (Espinoza et al., Education Sciences 15(7):827) ranked Newman's framework highest among five models, but that is an expert-judgment ranking, not a measured inter-rater agreement.

### Category 2: Algebra-Specific Misconception Taxonomies

#### Subcategory 2.1: The best-fit taxonomy (Booth)
**Source 1: Booth, Barbieri, Eyer and Pare-Blagoev, "Persistent and Pernicious Errors in Algebraic Problem Solving" (Journal of Problem Solving, 2014)**

DOK 1 - Facts
- The study coded the work of 565 Algebra I students (about 35% in grades 7 to 8) across six curricular topics including one-step, two-step, and multi-step equations.
- It used six conceptual error categories plus arithmetic: Variable errors, Negative-sign errors, Equality/inequality errors, Operation errors, Fraction errors, and Mathematical-property errors (for example misapplying the distributive property). Arithmetic errors were treated separately as attention or fact-recall rather than conceptual.
- Booth and Koedinger (2008) found equals-sign conceptual knowledge correlated with correct solutions at about R = .52 and negative-sign knowledge at about R = .48, and that these misconceptions predicted both lower performance and worse learning.

Summary and Analysis
This is the closest-matched, peer-reviewed scheme for the exact population this project targets, and its categories are defined by features visible in written work, which is what a classifier needs. It is the recommended backbone for the taxonomy. One important caveat determines how it should be used: the coding in Booth et al. (2014) was done by a single coder (the authors state this was to keep coding consistent across topics), so no inter-rater reliability figure exists for the scheme. The categories are grounded and well-matched, but their label boundaries have not been shown to reproduce across independent coders, which is a direct argument for keeping the label set coarse and for measuring agreement on a double-coded subset before trusting it.
Link to Source: Booth, Barbieri, Eyer and Pare-Blagoev (2014), Journal of Problem Solving 7(1). https://docs.lib.purdue.edu/jps/vol7/iss1/3/

#### Subcategory 2.2: The component misconceptions (variable, equals sign, conjoining)
**Source 1: Kuchemann (1981), variable interpretations, and Kieran (1981), the equals sign**

DOK 1 - Facts
- Kuchemann identified six ways students interpret algebraic letters, from least to most sophisticated: letter evaluated, letter ignored, letter as object, letter as specific unknown, letter as generalized number, letter as variable.
- In the CSMS data, a majority of 13-to-15-year-olds did not reach the "specific unknown" level (reported as 83% of 13-year-olds, 66% of 14-year-olds, 60% of 15-year-olds), and recent replications report similar figures.
- Kieran (1981) established the operational-versus-relational distinction for the equals sign: operational treats "=" as a "do something" or "here comes the answer" operator, relational treats it as a statement that both sides denote the same quantity.
- Knuth et al. (2006) linked relational understanding to success in solving equations after controlling for general achievement. Knuth et al. (2008) is often cited for a 58% operational versus 29% relational split among 6th and 7th graders.

Summary and Analysis
These sources supply the conceptual content behind two of Booth's families. Kuchemann's letter-as-object interpretation is the root of the conjoining error (writing 2 + 3a as 5a) and the reversal error, and it justifies treating "variable error" as one coherent bin. Kieran's operational-versus-relational distinction is what a "equality/balance error" actually measures: a student operating on one side only, or dropping the equals sign, is displaying operational thinking. Honesty flag: the specific percentages here (the Kuchemann level figures and especially the Knuth et al. 58%/29% split) come from particular populations and eras and could not all be independently verified in this research, so they should be treated as directional and checked before being cited as fixed facts.
Link to Source: Kuchemann (1981) in Hart (ed.), "Children's Understanding of Mathematics: 11-16"; Kieran (1981), Educational Studies in Mathematics 12(3); Knuth et al. (2006), Journal for Research in Mathematics Education 37(4).

### Category 3: Slips, Mistakes, and the Observability Ceiling

#### Subcategory 3.1: The slip-versus-mistake distinction
**Source 1: Norman (1981) and Reason (1990)**

DOK 1 - Facts
- Norman (1981), "Categorization of Action Slips," and Reason (1990), "Human Error," establish the canonical distinction: mistakes are errors of intention (the plan itself was wrong, rooted in faulty knowledge) while slips and lapses are errors of execution (the intention was right but the action or memory failed).
- In mathematics education this maps onto slips (random, careless, correctable when pointed out, not indicative of a misconception) versus conceptual errors (systematic, stable, arising from overgeneralization). Sources include Olivier (1996), Herholdt and Sapire (2014), and Ketterlin-Geller and Yovanoff (2009).

Summary and Analysis
This is the theoretical foundation for SPOV 3. The distinction that matters most pedagogically, whether a wrong answer reflects a stable misconception or a one-off slip, is defined by intention and stability, and neither is present in a single final answer. A student who writes x = 12 for x + 4 = 8 might hold a stable "add to both sides" misconception or might have misread a sign once. Distinguishing them requires either multiple problems or the shown work, and sometimes even the shown work is not enough. This is why the taxonomy needs a distinct "arithmetic slip" label and why the system needs an abstention path rather than a forced choice.
Link to Source: Norman (1981), Psychological Review 88(1); Reason (1990), "Human Error," Cambridge University Press.

#### Subcategory 3.2: Systematic bugs are real but unstable
**Source 1: Brown and Burton (1978); Brown and VanLehn (1980)**

DOK 1 - Facts
- Brown and Burton (1978) built the BUGGY and DEBUGGY systems, cataloguing systematic procedural bugs from thousands of students' multi-digit subtraction work. Errors are "systematic" when a consistent faulty procedure reproduces them.
- Brown and VanLehn (1980) repair theory explained how bugs arise: an incomplete procedure hits an impasse, the student applies a repair, and different repairs manifest as different bugs.
- They documented bug migration: bugs are unstable, and a student shifts among different bugs across problems while the underlying incomplete procedure stays the same.

Summary and Analysis
This work supports two things at once. It confirms that systematic procedural errors are real and catalogable, which is what makes fine-tuning on error types viable at all (Insight 4). But bug migration is a strong caution: because one stable underlying misconception can surface as several different observable errors, the mapping from what is on the page to the latent cause is many-to-many and noisy. That is an independent reason, separate from the slip-versus-mistake problem, to avoid claiming a clean one-to-one diagnosis and to build abstention into the design.
Link to Source: Brown and Burton (1978), Cognitive Science 2(2); Brown and VanLehn (1980), Cognitive Science 4(4).

### Category 4: Taxonomy Granularity and Reliability

#### Subcategory 4.1: Finer taxonomies reduce accuracy and agreement
**Source 1: Liu, Sonkar, Wang, Woodhead and Baraniuk (2023); Warrens (2010)**

DOK 1 - Facts
- Liu et al. (2023) evaluated GPT-4 on Eedi-derived misconception identification and reported accuracy falling from 91.9% when choosing among 4 candidate misconceptions to 39.8% when choosing among 100, an inverse relationship between the number of misconceptions and performance.
- Warrens (2010) proved formally that combining categories generally increases Cohen's kappa, so coarser schemes are structurally easier to agree on (though the effect is not universal in every case).
- Adjacent empirical work shows the same monotonic pattern on identical data (for example a coding study moving from 3-class to 5-class to 6-class schemes saw agreement fall from about 0.80 to 0.71 to 0.68).

Summary and Analysis
This is the quantitative core of SPOV 2. The Liu et al. curve is the single clearest demonstration that label count directly trades against accuracy, and the Warrens result gives the theoretical reason coarser schemes are easier to agree on. Together they argue that a coarse taxonomy is a design choice, not a limitation. Honesty flag: I did not find a study measuring the coarse-versus-fine reliability tradeoff inside a mathematics-error taxonomy specifically, so the claim rests on a formal proof plus adjacent-domain evidence plus the Liu et al. LLM result. It is well-supported but should be stated as a strong working assumption for this domain rather than a directly measured fact.
Link to Source: Liu et al. (2023), https://arxiv.org/abs/2310.02439 ; Warrens (2010), Statistical Methodology 7(6).

### Category 5: Recoverability and Machine Classification of Errors

#### Subcategory 5.1: What is recoverable from different levels of signal
**Source 1: McNichols, Zhang and Lan, "Algebra Error Classification with Large Language Models" (AIED, 2023)**

DOK 1 - Facts
- The task is framed exactly as this project frames it: given a student's step history, classify the error at a step.
- On a 24-label baseline scheme, BERT scored 80.68% versus tree-embedding and GRU baselines at 78.71% and 75.35% (about 3,318 steps).
- On the authors' own 19-label scheme, adding student-action context (an intent signal) raised BERT accuracy from 82.02% (equation-only) to 85.90% (with action context).
- Their confusion analysis found one class (REVERSED SIDES) misclassified as another (WRONG OPERATION) 41% of the time, because the true student intent is unknown and so it is genuinely ambiguous which label is correct.
- The authors state plainly that without perfect knowledge of student intent, certainty about the true cause of an error is impossible, and that this is a general limitation of all error-classification systems.

Summary and Analysis
This is the most directly relevant prior work and it does double duty. It sets a realistic accuracy bar (mid-80s with shown work on a multi-label scheme), which reframes the project's win as a base-versus-tuned delta on a coarse taxonomy rather than a capability contest. And its 82.0% to 85.9% jump quantifies that more observable signal raises the ceiling, while the permanent 41% confusion between two classes quantifies that the ceiling does not disappear. That residual ambiguity is exactly the region the abstention label is designed to catch.
Link to Source: https://arxiv.org/abs/2305.06163

#### Subcategory 5.2: Final-answer-only signal and distractor diagnosis
**Source 1: "Correct Answer Trap" analysis of reasoning-trace loss; de la Torre (2009) and MC-DINA distractor models**

DOK 1 - Facts
- Recent analysis of AI tutors ("Catching the Correct Answer Trap") finds that removing the reasoning trace removes the signal misconception classifiers rely on, and that verifying student reasoning step by step is hard for frontier models even when the final answer is correct.
- Cognitive diagnostic models built for multiple-choice items (de la Torre 2009; MC-DINA) map each distractor to a specific misconception, so which wrong option a student picks carries diagnostic information, but only when items are engineered with diagnostic distractors.
- Students with identical total scores can have different misconception profiles, meaning raw accuracy alone is insufficient to classify error type.

Summary and Analysis
This subcategory sets the lower and upper bounds on recoverability. On the low end, a final answer by itself is weakly diagnostic because many distinct errors collapse to the same number, which is why confidence should scale with how much work is visible. On the high end, a specific wrong answer can be strongly diagnostic, but only in the engineered multiple-choice case, which a general free-response equation is not. For this project, the practical consequence is that the dataset should include shown work wherever possible, and the model should abstain or lower its confidence on final-answer-only inputs rather than guess.
Link to Source: "Catching the Correct Answer Trap" (arXiv, 2026), https://arxiv.org/abs/2605.23925 ; de la Torre (2009), Applied Psychological Measurement 33(3).

---

## Candidate Taxonomy (working output, to be validated)

Grounded on Booth et al. (2014), trimmed to what is observable in one-variable linear-equation work, and kept deliberately coarse per SPOV 2. Seven labels:

1. **Equality/balance error** (operating on one side only, dropping the equals sign, not maintaining balance)
2. **Negative-sign error** (dropping or mishandling a negative, moving a term without flipping its sign)
3. **Variable error** (combining unlike terms, conjoining such as 2 + 3a = 5a, mishandling the coefficient)
4. **Operation/inverse error** (using the wrong inverse operation)
5. **Distribution/property error** (misapplying distributivity or order of operations)
6. **Arithmetic slip** (a computation error with no conceptual signature)
7. **Abstain / insufficient signal** (the calibrated abstention path for inputs where the observable does not distinguish among the above)

Validation plan tied to the SPOVs: start with these seven and do not go finer until agreement is measured; require a shown-work field and reserve confident labels for cases where the fingerprint is visible in a step; measure inter-rater reliability on a double-coded subset (target Cohen's kappa at or above 0.7) before adding any sub-categories, and merge any category whose per-label agreement is low rather than splitting it.

---

## Open Questions / To Verify
- Confirm the exact percentages attributed to Knuth et al. (2008) and Kuchemann (1981) against the primary sources before citing them as fixed figures.
- Confirm the arXiv identifier and publication status of the "Correct Answer Trap" analysis.
- Decide the output schema that carries confidence (a numeric score, a top-label-plus-runner-up, or a high/medium/low bucket), since this shapes every training example and the calibration step.
