# Kernel for the 'ICR - Identifying Age-Related Conditions' Competition

Check out my kernel on Kaggle.com: [Pub 0.29 Priv 0.48 Kernel for ICR (GLMNET, XGBM)](https://www.kaggle.com/code/hendriknebel/pub-0-29-priv-0-48-kernel-for-icr-glmnet-xgbm)

or check it out on GitHub

---

## My Summary Notes

### About the Competition
- see [ICR - Identifying Age Related Conditions](https://www.kaggle.com/competitions/icr-identify-age-related-conditions)

---

### Particularities of the Competition

##### Test Sets (1) [aka 'public'] & (2) [aka 'private'] are unknown
- During the competition, the test set could not be displayed.

##### Final results computed from Test Set (2)
- During the competition, test runs were possible on the 1st half of the entire Test Set.
- The final results were computed on the 2nd half of the entire Test Set.

##### Testing is scarce
- The competition participants were only to run/test 1 model per day. (Other competitions usually allow for up to 5 trials per day.)

##### Test metric penalizes confident, wrong predictions
- A test run only returns one metric. Here, the evaluation metric is computed by a '*balanced* log loss' function (see competition page).
- The predictions were to be made as probabilities (e.g. 0.89, 0.11), not as class labels ('0' or '1').
- A confident prediction for a class label ('The predicted probability is high.') that is wrong, leads to a huge drop in the final evaluation metric.
- A model should take that into account, and predict approx. (0.5, 0.5) (coin flip) when in doubt.

##### Models trained on the Training Set perform (far) worse on the Test Set (1) & Test Set (2)
- With all that, models trained & cross-validated on the Training Set generally perform bad on the Test Set. Overfitting was a general issue to deal with.
- *A low CV mean training error & low CV training error spread does not say a thing, unless the model is run on Test Set giving similar results.*

---

### Strategic Approach

##### Get an overview
- With a limited number of test trials, an unknown test set, and overfitting as a main general issue, the task of classifying data points gets even harder.

##### Test many different models
- In order to counter overfitting, a possible strategy is to test many single ML models one-by-one. A drawdown with this approach is that it wastes a lot of trials. However, it's beneficial to establish which models perform at all & which are useless.

##### Decide on which models to test further
- Based on the single tests, one can decide on which model to work on further. It's also possible to combine/ensemble different models.

##### Do not shy away from trying specialized/smooth models
- Many approaches tried to find specialized models as far as the training & test set allowed.
- Many approaches used probing techniques on the test set. Though not beneficial to the question, which model describes the data best, these approaches were allowed. Taking a look at the difference between the performance of these approaches on the Test Set (1) & Test Set (2), it becomes clear that they are discouraged.

##### Generalize/regularize well
- The only one strategy in this case (strong overfitting) is choosing generalized/regularized approaches.
- That is, the model is allowed to make more errors to not fit the data as perfectly as before.
- One takes into account a larger test error in return for greater consistency in predictions.

##### Choose your submissions
- During the competition, the best performing approaches reached eval metrics of 0.06 and below. Many approaches favored post-processing. A large part of very well performing models centered around approaches with an eval metric of about 0.11. Better performing models in general were around 0.20 or smaller.
- A model that performed with an eval metric of around 0.69 made (0.50, 0.50) predictions for all data points, basically.
- I found the approaches with a test balanced log loss > 0.11 and < 0.3 the most convincing. Though in doubt, I decided on submitting one approach in the middle, and one more generalized approach.

##### Final Results
- Due to the particularities of the competition, a (huge) shake-up from test set (1) evaluation to the final evaluation on the test set (2) was to be expected. - (No) Surprise!: There was indeed a huge shake-up among all the participants!
- Should have decided on and should have worked on approaches that generalize well even more!
