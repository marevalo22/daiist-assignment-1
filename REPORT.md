# Assignment 1 Report

- **Name**: Matias Arevalo
- **Student ID**:
- **Email**:
- **Group**: [BBADBA 5A]

## Dataset

I used the Speed Dating Experiment data from Kaggle. Fisman, Iyengar, Kamenica and Simonson collected it at Columbia Business School between October 2002 and April 2004 and published the study in the Quarterly Journal of Economics in 2006. Students signed up for speed-dating nights ("waves") and filled in a survey a few days before. On the night they met every person of the opposite sex for four minutes. After each date they ticked yes or no and rated the partner on six attributes.

The raw file has 8,378 rows and 195 columns. One row is one date seen from one person's side, so every date is in the file twice. There are 551 people (274 women and 277 men) across 21 events, and each person meets between 5 and 22 partners. The data key sorts the columns by when they were collected: the sign-up survey, the scorecard filled in during the night, and follow-up surveys sent afterwards.

The pipeline (`train.py`, with the shared code in `speeddating/`) starts from this raw file and nothing else. The Colab notebook `assignment1_matias_arevalo.ipynb` runs the same steps in one file. It joins each woman's record of a date with the man's record of the same date, which gives 4,184 woman-man pairs. Ten records have no partner id and cannot be paired.

I chose this dataset for three reasons. The target has to be built, because a match needs two decisions that sit in two different rows. Most of the columns that look predictive are filled in during the date, so I had to decide what a model may see. And the rows are grouped by person and by event, so the split needs thought. In the end these three questions mattered more than the choice of model.

## Business / real-life framing

### Client and decision

The client is a company that runs speed-dating nights. Today everyone meets everyone. With 22 people per side that is 22 four-minute dates plus changeovers, close to two hours. The organiser wants to run bigger or shorter nights where each person meets only part of the room. The decision is made the day before: which pairs sit together? At that point the organiser has the sign-up forms and the guest list. The model gives, for every possible woman-man pair at an upcoming event, the probability that both will say yes.

### What the framing decides in the pipeline

1. Target. The target is the mutual match, `dec_W * dec_M`, which is 1 for 16.5% of pairs. A single yes is worth nothing to the organiser, because contact details are only exchanged when both say yes. The pipeline checks that this product equals the raw `match` column on every pair.
2. Inputs. The model only gets what exists before the night. The scorecard ratings (attractive, sincere, intelligent, fun, ambitious, shared interests and "how much do you like this person") are the strongest predictors in the file. People write them during the date, after the conversation that decides the outcome, so using them would be target leakage. I kept them out of the model and used them only in a separate "what if" analysis (section 8b of `train.py`).
3. Split. Whole events go to one side, in date order. Waves 1-13 are training (2,507 pairs), waves 14-16 validation (750 pairs) and waves 17-21 test (927 pairs, February to April 2004). The organiser always scores a future night full of new people. A random split would put the same people on both sides: each person appears in about 15 pairs, and in a 5-fold random split every evaluation pair involves someone the model has already seen. On the development events the same model scored a cross-validated AUC of 0.628 with random rows and 0.599 with whole events held out. Date order matters as well because the study changed over time. Waves 6-9 used a different preference scale, wave 12 limited how many people you could say yes to, and in waves 18-21 people were asked to bring a book or magazine.
4. Threshold. Predicting "no match" for every pair is 84.8% accurate on the test events and tells the organiser nothing, so I priced the two errors instead. I assume a ticket of about EUR 25 for 10-12 dates. A four-minute slot is then worth about EUR 2.50 per person, or EUR 5 for the two people at the table. That is the cost of seating a pair that does not match (a false positive). A match is what people pay for and what brings them back, so missing one costs more. I assume EUR 20, roughly one lost repeat ticket (a false negative). Seating a pair pays when p x 20 > (1 - p) x 5, i.e. when p > 5 / (5 + 20) = 0.20. This rule only works if the probabilities are calibrated, which is why I tuned the model on log loss. AUC and accuracy say nothing about calibration. Both costs are my assumptions, and the dashboard lets you change them.
5. A second evaluation, closer to the real decision. A threshold judges pairs one at a time, but a night has a hard constraint: every person gets the same number of dates. So I also simulate the night. For each test event, each person may meet only a share of the other side (30%, 50% or 70%). The model's plan is the set of dates with the highest total predicted probability under that constraint, found with a small linear program. I count how many real matches the plan keeps and compare it with random plans of the same size and with a perfect plan.

![Match rate by event and split](figures/target_by_wave.png)

## Data preparation & feature engineering

### Cleaning decisions

Every cleaning step in the pipeline comes from a check on the raw data that `train.py` runs and logs, and no step is undone later.

- Two question blocks are dropped. Block `4_1` ("what do you think your own sex looks for") was not asked in waves 1-5, and block `5_1` ("how do you think others rate you") was not asked in waves 1-9. The training events would have to be filled almost entirely by imputation, so I did not use them. Every other sign-up block is missing for 2% of people or fewer.
- Missing answers stay missing until the model pipeline fills them. The imputer uses the median of the training rows only, inside an sklearn `Pipeline`, and the same holds inside every cross-validation fold. Nothing is imputed or scaled on the full data.
- Seven people (5 women, 2 men) never filled in the sign-up survey. Their answers are imputed like the rest, and a `survey_missing` flag lets the model treat them separately.
- `art` is dropped from the interests. Its correlation with `museums` is 0.87 (measured once per person), the only pair above 0.8. Keeping both would count that taste twice in the interest features.
- The six "what do you look for" answers are rescaled so each person's weights sum to 100. Waves 6-9 first rated each item 1-10. The published file already turns those ratings into shares, which is why 78-100% of the people in those waves have decimals. A few people in other waves still have totals between 90 and 148. Rescaling fixes both, and I kept the decimals because rounding to whole numbers would distort waves 6-9 the most.
- Event size comes from the women's `round` column, the number of men at the event. When the two sides are not the same size, `round` differs between a woman's record and a man's record of the same date.

![Missing sign-up answers](figures/eda_missing.png)

### Features (40, all known before the night)

One row is a pair, so some features describe one person and others describe the combination.

Profile features, per person (14 x 2 = 28). Age. How often they date and how often they go out (1 = several times a week, 7 = almost never, kept as ordinal). Expected happiness with the night. Three indicators for the main goal: serious relationship, get a date, fun night out. Goal is a category, so treating its codes 1-6 as a number would be wrong; I kept the three answers that differ in intent and left the others as the baseline. Then the five self-ratings (attractive, sincere, intelligent, fun, ambitious), the share of the 100 preference points given to looks, and the missing-survey flag.

Pair features (11). These are the reason to work at pair level.

- `pref_fit_W` and `pref_fit_M`: what she says she values, weighted against how he rates himself on those attributes, and the reverse. A woman who gives 40 points to "intelligent" paired with a man who rates his intelligence 9 scores high.
- `interest_corr` and `interest_gap`: the correlation and the mean absolute difference of the two people's 12 interest ratings. The correlation measures whether they like the same things, the gap whether they like them equally strongly.
- `age_gap` (his age minus hers), `age_gap_abs`, `same_field`, and `looks_gap` (his self-rated looks minus hers).
- `same_race` and `race_mismatch_W/M` = (1 - same race) x how important same race is to that person (1-10). A different race only matters to people who say it matters, and a linear model cannot learn that interaction unless I build it. I did not use race categories as inputs. The organiser would then be deciding who meets whom based on ethnicity, and the stated preference already carries the useful part.

Event feature (1): the number of men at the event, which is also the number of partners each woman meets.

### Does each block earn its place?

Same model, same C, leave-one-event-out over the 16 development events:

| setting | features | log loss | AUC | mean AUC within an event |
|---|---|---|---|---|
| all 40 features | 40 | 0.4456 | 0.602 | 0.594 |
| without profile features | 12 | 0.4490 | 0.575 | 0.580 |
| without pair features | 29 | 0.4484 | 0.585 | 0.574 |
| without event size | 39 | 0.4456 | 0.603 | 0.594 |

Dropping the pair features hurts the within-event AUC most, and ranking pairs inside one night is what the organiser needs. Event size adds nothing measurable. I kept it because the organiser controls it, and the penalty shrinks its coefficient anyway.

### Tried and dropped

I added each of these blocks on top of the 40 features and scored it the same way. All four made log loss worse, so I kept none of them:

| block added | features | log loss | AUC | mean AUC within an event |
|---|---|---|---|---|
| raw interest levels (12 per person) | 64 | 0.4487 | 0.586 | 0.586 |
| all six preference weights per person | 50 | 0.4470 | 0.595 | 0.598 |
| field of study, 6 groups per person | 52 | 0.4506 | 0.585 | 0.584 |
| beliefs about what the other side wants | 52 | 0.4463 | 0.601 | 0.590 |

The raw interests show the problem most clearly. The 24 extra columns describe each person on their own, and with about 550 positive examples they mostly add noise. The two pair-level summaries of the same interests (`interest_corr` and `interest_gap`) carry what is useful in two columns.

## Modeling

### Model and objective

The model is logistic regression with an L2 penalty: the target is binary, and the decision rule needs probabilities. All three implementations minimise the same objective on standardised features:

J(w, b) = mean binary cross-entropy + (lambda / 2) ||w||^2, with lambda = 1 / (C n)

scikit-learn writes it as C x (sum of losses) + ||w||^2 / 2. Dividing by C n gives the line above. None of the three penalises the intercept. In the PyTorch versions I add the penalty to the loss by hand. The optimiser's `weight_decay` would also shrink the bias, and then the three would no longer solve the same problem.

### Regularisation strength C (shared by all three)

I chose C by leave-one-event-out log loss on the 16 development events, because a single split with three validation events was too noisy to choose from. Log loss was 0.4500 at C = 0.0003, 0.4465 at 0.001, 0.4456 at 0.003, 0.4476 at 0.01, 0.4500 at 0.03, 0.4517 at 0.1 and 0.4525 at 1. C = 0.003 won. That is strong regularisation (lambda is about 0.1), which fits a weak signal and few positives per feature.

### Learning rates (PyTorch only)

I fitted both PyTorch versions on waves 1-13 for 1,500 full-batch epochs. The learning rate only controls how the optimiser travels, and its job is to reach the minimum of the training objective. So I picked the rate with the lowest final training objective that did not diverge, and reported validation log loss next to it. I did not pick on validation loss. A learning rate that is too small stops before the minimum, which acts like extra regularisation, and I would then be comparing a different model to sklearn without noticing. The search shows this: the manual loop at lr = 0.01 had the best validation log loss of all (0.4731 against 0.4757) only because it had not converged. Plain gradient descent diverged at lr = 4.0.

The `nn.Module` version uses Adam with a cosine schedule that lowers the step size to 0 by the last epoch. I added the schedule after a first run in Colab without it. With a constant rate, Adam ended about 7e-7 above the minimum on the development events, its coefficients differed from sklearn's by up to 0.002, and it took a different seat-or-not decision on one test pair. Adam divides each step by the recent size of the gradient, so near the minimum its steps stop shrinking and it keeps circling. Plain gradient descent does not have this problem, because its step is proportional to the gradient, which goes to zero at the minimum. With the schedule, Adam at lr = 0.05 and 0.1 ended within 1e-14 of sklearn's minimum on the training events, lr = 0.01 stopped 1e-9 above it, and lr = 0.001 had not converged (0.09 above). Chosen: manual loop lr = 2.0, `nn.Module` + Adam lr = 0.1.

### Final fit and test results

All three were refitted on the 16 development events (3,257 pairs) with the same preprocessing and scored once on the 927 test pairs (141 matches). Costs use the EUR 5 and EUR 20 assumptions and the 0.20 threshold.

| model | ROC AUC | mean AUC within an event | PR AUC | log loss | Brier | cost at t = 0.20 |
|---|---|---|---|---|---|---|
| scikit-learn `LogisticRegression` (lbfgs) | 0.558 | 0.572 | 0.226 | 0.4246 | 0.1281 | EUR 2,745 |
| PyTorch, manual loop (gradient descent) | 0.558 | 0.572 | 0.226 | 0.4246 | 0.1281 | EUR 2,745 |
| PyTorch, `nn.Module` + Adam | 0.558 | 0.572 | 0.226 | 0.4246 | 0.1281 | EUR 2,745 |
| baseline: training match rate for every pair (seat nobody) | 0.500 | n/a | 0.152 | 0.4273 | 0.1292 | EUR 2,820 |
| baseline: seat everyone (today's full rotation) | 0.500 | n/a | 0.152 | n/a | n/a | EUR 3,930 |

### Do the three agree?

Yes. The largest coefficient difference is 7.2e-08 between sklearn and the manual loop and 6.9e-08 between sklearn and `nn.Module`. The intercepts differ by at most 2.4e-07, the largest difference in any test probability is 1.5e-07, and all 927 seat-or-not decisions are identical. The training objective at the final weights is 0.43813989 for all three. The objective is convex, so it has one minimum, and three correct implementations of it should find the same point. The small differences that remain come from precision and stopping: PyTorch works in float32 (about 7 significant digits) and sklearn in float64, and each optimiser stops at a slightly different point near the minimum. Fitting took 0.01 s for sklearn (L-BFGS) and under a second for each PyTorch version (1,500 epochs each).

![Three implementations](figures/three_implementations.png)

### What the results mean for the organiser

The model is weak. The test AUC of 0.558 has a bootstrap 95% interval of 0.508 to 0.614. Ranking inside one woman's list of partners works better (mean AUC 0.612) than inside one man's list (0.535). The probabilities are compressed by the strong penalty: the bin averages in the calibration plot run from 0.10 to 0.26. They are roughly calibrated except in the lowest bin, where the real match rate (0.16) is above the prediction (0.10). On the validation events the cost was lowest at a threshold of 0.165, close to the 0.20 the cost rule gives.

In money, at t = 0.20 the model seats 205 of the 927 test pairs (22%), catches 44 of the 141 matches and costs EUR 2,745. Seating nobody costs EUR 2,820 and seating everyone EUR 3,930. The saving is EUR 75 (2.7%) over five events: real, but small, and within the noise of five test nights. It also depends on the cost ratio. If a missed match costs EUR 40 instead of 20, the threshold drops to 0.11, and seating everyone (EUR 3,930) becomes cheaper than following the model (EUR 4,260), because the probabilities are too compressed to separate pairs at that level. The model is useful when seats are scarce, which is the seating-plan case below.

![Calibration and business cost](figures/calibration_and_cost.png)

The seating simulation shows the effect more clearly. When each person meets half of the room, the model's plans keep 81 of the 141 test matches. Random plans with the same number of dates keep 72 on average (sd 4.9), and only 5% of 200 random plans do as well as the model. A perfect plan would keep 138. With 30% of the room the model keeps 57 against 45 for random plans (1% of them as good). With 70% it keeps 108 against 102 (12% as good). So the model adds about 6 to 12 matches over five events when the night is cut short, and the gap to the perfect plan is much larger than that gain.

![Seating plans](figures/planner.png)

What the model learned (standardised coefficients, comparable with each other): men who rate themselves as fun, and men who go out and date often, are matched more. Older men and larger age gaps are matched less. A different race combined with saying race matters lowers the chance, on both sides. Women who expect to be happy with the night, pairs from the same field, and pairs where he scores well on what she says she values do better.

## Limitations & next steps

### The missing piece: physical attraction

The most important input is not in the sign-up data. How attractive two people find each other is one of the strongest drivers of a yes on the night, and before the night the only looks signal is a self-rating. Five numbers from the data show how large that gap is:

1. Self-rated attractiveness barely tracks how attractive others found the person. The correlation between the sign-up self-rating and the average rating received on the night is 0.29 (0.17 for women, 0.36 for men).
2. The rating a person receives is strongly tied to being chosen. The correlation between a person's average received attractiveness and the share of partners who said yes to them is 0.79.
3. At pair level, his yes correlates 0.075 with her self-rated looks and 0.516 with how attractive he found her. For her yes the numbers are 0.088 and 0.445.
4. When I train the same model on one side's decision, her yes is somewhat predictable from the sign-up form (test AUC 0.654) and his yes is close to a coin flip (0.526). This fits the original study's finding that men weigh looks more heavily than women do, and looks are what the form does not capture. A mutual match needs both decisions, so the weaker side limits the whole model.
5. I added information step by step (sklearn, C re-tuned the same way, same test events):

| information | test ROC AUC | matches kept, half-room plan |
|---|---|---|
| sign-up data only (the deployable model) | 0.558 | 81 |
| + "looks reputation": average attractiveness score from everyone else met that night | 0.636 | 81 |
| + how attractive the two found each other | 0.739 | 111 |
| + the full scorecard | 0.805 | 115 |

Rows 2-4 use information from the night itself, so the organiser does not have them; they measure the size of the gap. Row 2 is the interesting one. A looks reputation is roughly what a photo rating would give, and it raises AUC, yet it keeps no extra matches in the seating plan. It tells you who is popular in general, and a fair plan gives everyone the same number of dates regardless. Only pairwise attraction (row 3) moves the plan, from 81 to 111 matches. Joel, Eastwick and Finkel (2017, Psychological Science) found something similar in a different speed-dating sample: questionnaire answers collected beforehand predicted how much a person desired others in general and how much they were desired, but not which specific pairs would click.

![The attraction gap](figures/attraction_gap.png)

What I would do about it: ask for photos at sign-up and have other members rate them, or reuse ratings from a person's earlier events, then add mutual-preference features such as "her stated type x his rated looks". Row 2 suggests that a popularity score alone will not help the seating plan; the model needs the pairwise part.

### Size of the data

The file looks larger than it is. The 8,378 rows are 4,184 pairs, but only 551 people and 21 events, and each person appears in about 15 pairs, so the rows are far from independent.

- There are 690 matches in total and 549 in the 16 development events, about 13.7 positives per feature. The usual rule of thumb for logistic regression is 10-20, so 40 features is about the limit. This is why the C search chose strong regularisation, and why every extra block of features in the table above made log loss worse.
- The test set has 927 pairs, 141 matches and five events. The bootstrap interval for the test AUC (0.508 to 0.614) is optimistic, because it resamples pairs as if they were independent. Differences below about 0.05 AUC between two feature sets cannot be told apart on this test set, and the EUR 75 saving is within that noise.
- The learning curve trains on 2 to 13 randomly drawn training events (40 draws per size) and scores on the validation events. Validation AUC rises from 0.537 with two events to 0.578 with 13, about +0.01 per doubling of the data from six events up, while training AUC falls towards it. More events would help, slowly. If that trend held, going from 0.58 to 0.65 would take around a hundred times more events, so more rows of the same survey will not close the gap on their own.
- The sample is students at one university in 2002-2004, and only woman-man pairs. Nothing here shows that the model transfers to another city, another age group or same-sex events.

![Learning curve](figures/learning_curve.png)

### Other limitations

- The costs (EUR 5 and EUR 20) are my assumptions. The threshold moves with them, which is why the dashboard lets you change both.
- The experiment changed between nights (the limit on yeses in wave 12, the book and magazine in waves 18-21). I kept all events. With more time I would test an indicator for these conditions, although with 21 events each one would be estimated from one to four nights.
- The sign-up survey is self-reported, and stated preferences are known to differ from how people choose on the night.
- The seating simulation assumes everyone gets the same number of dates and that the outcome of a date does not depend on who else that person met that night. Real nights have order and fatigue effects.

## Generative AI use disclosure

I used a mix of Claude (Anthropic) and ChatGPT (OpenAI) during this assignment. 
