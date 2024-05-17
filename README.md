Disaster Relief Project
================
Alex Link

-   <a href="#function-coding" id="toc-function-coding">Function Coding</a>
-   <a href="#k-folds-out-of-sampling-performance"
    id="toc-k-folds-out-of-sampling-performance"><strong>K-Folds Out of
    Sampling Performance</strong></a>
    -   <a
        href="#underlinedataspace-readingspace-cleaningspace-andspace-shuffling"
        id="toc-underlinedataspace-readingspace-cleaningspace-andspace-shuffling"><span
        class="math inline">$\underline{Data\space Reading,\space Cleaning\space
        and\space Shuffling}$</span></a>
    -   <a href="#underlineknn" id="toc-underlineknn"><span
        class="math inline">$\underline{KNN}$</span></a>
        -   <a href="#knn-roc-curve" id="toc-knn-roc-curve">KNN ROC Curve</a>
        -   <a href="#knn-precision-recall-curve"
            id="toc-knn-precision-recall-curve">KNN Precision Recall Curve</a>
        -   <a href="#knn-confusion-matrix" id="toc-knn-confusion-matrix">KNN
            Confusion Matrix</a>
        -   <a href="#knn-threshold-rationale" id="toc-knn-threshold-rationale">KNN
            Threshold Rationale</a>
    -   <a href="#underlinelda" id="toc-underlinelda"><span
        class="math inline">$\underline{LDA}$</span></a>
        -   <a href="#lda-roc-curve" id="toc-lda-roc-curve">LDA ROC Curve</a>
        -   <a href="#lda-precision-recall-curve"
            id="toc-lda-precision-recall-curve">LDA Precision Recall Curve</a>
        -   <a href="#lda-confusion-matrix" id="toc-lda-confusion-matrix">LDA
            Confusion Matrix</a>
        -   <a href="#lda-threshold-rationale" id="toc-lda-threshold-rationale">LDA
            Threshold Rationale</a>
    -   <a href="#underlineqda" id="toc-underlineqda"><span
        class="math inline">$\underline{QDA}$</span></a>
        -   <a href="#qda-roc-curve" id="toc-qda-roc-curve">QDA ROC Curve</a>
        -   <a href="#qda-precision-recall-curve"
            id="toc-qda-precision-recall-curve">QDA Precision Recall Curve</a>
        -   <a href="#qda-confusion-matrix" id="toc-qda-confusion-matrix">QDA
            Confusion Matrix</a>
        -   <a href="#qda-threshold-rationale" id="toc-qda-threshold-rationale">QDA
            Threshold Rationale</a>
    -   <a href="#underlinelogisticspace-regression"
        id="toc-underlinelogisticspace-regression"><span
        class="math inline">$\underline{Logistic\space Regression}$</span></a>
        -   <a href="#logistic-regression-roc-curve"
            id="toc-logistic-regression-roc-curve">Logistic Regression ROC Curve</a>
        -   <a href="#logistic-regression-precision-recall-curve"
            id="toc-logistic-regression-precision-recall-curve">Logistic Regression
            Precision Recall Curve</a>
        -   <a href="#logistic-regression-confusion-matrix"
            id="toc-logistic-regression-confusion-matrix">Logistic Regression
            Confusion Matrix</a>
        -   <a href="#logistic-regression-threshold-rationale"
            id="toc-logistic-regression-threshold-rationale">Logistic Regression
            Threshold Rationale</a>
    -   <a href="#underlinerandomspace-forest"
        id="toc-underlinerandomspace-forest"><span
        class="math inline">$\underline{Random\space Forest}$</span></a>
        -   <a href="#random-forest-tuning-parameter-interpretation--explanation"
            id="toc-random-forest-tuning-parameter-interpretation--explanation">Random
            Forest Tuning Parameter Interpretation &amp; Explanation</a>
        -   <a href="#random-forest-roc-curve"
            id="toc-random-forest-roc-curve">Random Forest ROC Curve</a>
        -   <a href="#random-forest-precision-recall-curve"
            id="toc-random-forest-precision-recall-curve">Random Forest Precision
            Recall Curve</a>
        -   <a href="#random-forest-confusion-matrix"
            id="toc-random-forest-confusion-matrix">Random Forest Confusion
            Matrix</a>
        -   <a href="#random-forest-threshold-rationale"
            id="toc-random-forest-threshold-rationale">Random Forest Threshold
            Rationale</a>
    -   <a href="#underlinesvm" id="toc-underlinesvm"><span
        class="math inline">$\underline{SVM}$</span></a>
        -   <a href="#svm-tuning-parameter-interpretation--explanation"
            id="toc-svm-tuning-parameter-interpretation--explanation">SVM Tuning
            Parameter Interpretation &amp; Explanation</a>
        -   <a href="#svm-roc-curve" id="toc-svm-roc-curve">SVM ROC Curve</a>
        -   <a href="#svm-precision-recall-curve"
            id="toc-svm-precision-recall-curve">SVM Precision Recall Curve</a>
        -   <a href="#svm-confusion-matrix" id="toc-svm-confusion-matrix">SVM
            Confusion Matrix</a>
        -   <a href="#svm-threshold-rationale" id="toc-svm-threshold-rationale">SVM
            Threshold Rationale</a>
    -   <a href="#k-folds-out-of-sampling-performance-table"
        id="toc-k-folds-out-of-sampling-performance-table"><strong>K-Folds Out
        of Sampling Performance Table</strong></a>
-   <a href="#hold-out-sample-performance"
    id="toc-hold-out-sample-performance"><strong>Hold-Out Sample
    Performance</strong></a>
    -   <a href="#underlinedataspace-readingspace-andspace-cleaning"
        id="toc-underlinedataspace-readingspace-andspace-cleaning"><span
        class="math inline">$\underline{Data\space Reading\space and\space
        Cleaning}$</span></a>
    -   <a href="#underlineknn-1" id="toc-underlineknn-1"><span
        class="math inline">$\underline{KNN}$</span></a>
    -   <a href="#underlinelda-1" id="toc-underlinelda-1"><span
        class="math inline">$\underline{LDA}$</span></a>
    -   <a href="#underlineqda-1" id="toc-underlineqda-1"><span
        class="math inline">$\underline{QDA}$</span></a>
    -   <a href="#underlinelogisticspace-regression-1"
        id="toc-underlinelogisticspace-regression-1"><span
        class="math inline">$\underline{Logistic\space Regression}$</span></a>
    -   <a href="#underlinerandomspace-forest-1"
        id="toc-underlinerandomspace-forest-1"><span
        class="math inline">$\underline{Random\space Forest}$</span></a>
    -   <a href="#underlinesvm-1" id="toc-underlinesvm-1"><span
        class="math inline">$\underline{SVM}$</span></a>
    -   <a href="#hold-out-sample-performance-table"
        id="toc-hold-out-sample-performance-table"><strong>Hold-Out Sample
        Performance Table</strong></a>
-   <a href="#conclusions"
    id="toc-conclusions"><strong>Conclusions</strong></a>
    -   <a
        href="#1-discussion-of-best-performing-algorithms-in-the-cross-validation-and-hold-out-data"
        id="toc-1-discussion-of-best-performing-algorithms-in-the-cross-validation-and-hold-out-data">1.
        Discussion of Best Performing Algorithm(s) in the Cross-Validation and
        Hold-Out Data</a>
    -   <a
        href="#2-discussionanalysis-justifying-why-findings-above-are-compatible-or-reconcilable"
        id="toc-2-discussionanalysis-justifying-why-findings-above-are-compatible-or-reconcilable">2.
        Discussion/Analysis Justifying Why Findings Above are Compatible or
        Reconcilable</a>
    -   <a
        href="#3-recommendation--rationale-regarding-chosen-algorithm-for-detection-of-blue-tarps"
        id="toc-3-recommendation--rationale-regarding-chosen-algorithm-for-detection-of-blue-tarps">3.
        Recommendation &amp; Rationale Regarding Chosen Algorithm For Detection
        of Blue Tarps</a>
    -   <a
        href="#4-discussion-of-the-relevance-of-the-metrics-calculated-in-the-tables-for-this-context"
        id="toc-4-discussion-of-the-relevance-of-the-metrics-calculated-in-the-tables-for-this-context">4.
        Discussion of the Relevance of the Metrics Calculated in the Tables for
        this Context</a>
    -   <a href="#5-why-precision-recall-over-roc-curves"
        id="toc-5-why-precision-recall-over-roc-curves">5. Why Precision Recall
        over ROC Curves?</a>
    -   <a href="#6-suggested-improvement-to-training-methoddata"
        id="toc-6-suggested-improvement-to-training-methoddata">6. Suggested
        Improvement to Training Method/Data</a>

###### Function Coding

### **K-Folds Out of Sampling Performance**

#### $\underline{Data\space Reading,\space Cleaning\space and\space Shuffling}$

#### $\underline{KNN}$

![](/images/knnTuning.png)

##### KNN ROC Curve

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-7-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-7-2.png" width="60%" style="display: block; margin: auto;" />

##### KNN Precision Recall Curve

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-8-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-8-2.png" width="60%" style="display: block; margin: auto;" />

##### KNN Confusion Matrix

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-9-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-9-2.png" width="60%" style="display: block; margin: auto;" />

##### KNN Threshold Rationale

The threshsold of 0.062 was chosen based off the mean Precision Recall
Curve, as well as my preference for a high sensitivity/recall. While a
threshold of 0.062 does provide a high accuracy, there are other
threshold values that provide higher overall accuracy. However, my main
goal for this problem would be locating and providing food and water to
as many displaced people as possible, irregardless of the resources at
my disposal; I would want to ensure as few displaced people are
overlooked as possible. This corresponds to a higher sensitivity/recall,
and thus lower false negative rate (predicted No Tarp, but there
actually is a Blue Tarp/Displaced Person there). Based on the mean
Precision Recall Curve formulated using the out-of-folds sample data, a
threshold of 0.062 has a near perfect recall of 1 (over 0.99), which
comes at a cost of precision, which drops to approximately 0.80.
However, given my previously stated rationale, I would prefer to
“overpredict” that there is a Blue Tarp/Displaced person, to ensure a
lower false negative rate & make sure as few displaced people are
overlooked as possible. Although this would lead to more predicitions of
Blue Tarp, which would incur more time and resources to send help to
those locations (even if there is not a displaced person located there),
I believe these possible sunk costs are outweighed by the lower false
negative rate, as it is more human lives saved; call me a hippy, but I
don’t believe in placing a value on human life, even though some
government agency seem to do so
(<https://www.transportation.gov/sites/dot.gov/files/docs/2016%20Revised%20Value%20of%20a%20Statistical%20Life%20Guidance.pdf>).
Applying this threshold to the out-of-folds sample data resulted in an
accuracy of over 99%, but most importantly, almost accomplished my goal
of providing aid to all displaced persons. Unfortunately 15 people (less
than 1% of all displaced people) were misclassified (assuming all blue
tarps correspond to displaced people) for this specific data set, but
the sensitivity/recall was still very good, indicating many people would
be located and (hopefully) be provided with aid (especially compared to
Puerto Rico-Hurricane Maria standards).

#### $\underline{LDA}$

##### LDA ROC Curve

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-11-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-11-2.png" width="60%" style="display: block; margin: auto;" />

##### LDA Precision Recall Curve

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-12-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-12-2.png" width="60%" style="display: block; margin: auto;" />

##### LDA Confusion Matrix

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-13-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-13-2.png" width="60%" style="display: block; margin: auto;" />

##### LDA Threshold Rationale

As stated above, my main goal for this problem is locating and providing
food and water to as many displaced people as possible, so keeping this
in mind, my threshold selection again will be quite low. However, when
compared to KNN, the Precision Recall AUC estimates for LDA are much
lower, and vary much more across folds (standard deviation of AUC
estimates across folds is approximately 6 times larger for LDA than
KNN). Based off the mean Precision Recall Curve, it appears I reach a
0.99 sensitvity on the out-of-folds sample data with a threshold of
1.88^{-4}. Unfortunately, this comes at a MUCH greater cost of precision
compared to KNN, as it falls to nearly 0.20, and although I previously
stated I am not concerned with misclassifying a non-blue tarp as a blue
tarp, dropping the threshold too low could end up making the model
almost useless (predicts Blue Tarp every time) or in this case, it would
be hard for me to ignore my previous assertion of not being concerned
with time and resources, as LDA predicts Blue Tarp approximately 4 times
as much as KNN (producing over 16 times as many false negatives!!), and
even manages to have a higher false negative rate, overlooking 5 more
displaced persons compared to KNN. All other relevant metrics also
underperformed compared to KNN.

#### $\underline{QDA}$

##### QDA ROC Curve

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-15-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-15-2.png" width="60%" style="display: block; margin: auto;" />

##### QDA Precision Recall Curve

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-16-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-16-2.png" width="60%" style="display: block; margin: auto;" />

##### QDA Confusion Matrix

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-17-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-17-2.png" width="60%" style="display: block; margin: auto;" />

##### QDA Threshold Rationale

Sticking with the theme of desiring a higher sensitivity/recall (at the
cost of precision), the mean Precision Recall Curve for this method gets
close to a sensitivity/recall of 1 (0.992 to be more precise), without
giving up as much precision (when compared to LDA), on the out-of-folds
sample data with a threshold of 0.021. Applying this threshold to the
out-of-folds sample data results in a sensitivty nearly on par with the
KNN method (has 1 more false negative than the KNN method), but it also
has more uncertainty/variability in this metric as evident by its higher
standard deviation. It also underperforms the KNN method in nearly every
other metric, and in fact misclassifies nearly 3 times as many non-blue
tarp images as blue tarp.

#### $\underline{Logistic\space Regression}$

##### Logistic Regression ROC Curve

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-19-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-19-2.png" width="60%" style="display: block; margin: auto;" />

##### Logistic Regression Precision Recall Curve

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-20-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-20-2.png" width="60%" style="display: block; margin: auto;" />

##### Logistic Regression Confusion Matrix

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-21-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-21-2.png" width="60%" style="display: block; margin: auto;" />

##### Logistic Regression Threshold Rationale

Again examining the mean Precision Recall Curve on the out-of-folds
sample data, it appears we can get very close to a sensitivity/recall of
1 (over 0.99) at a threshold of 0.0135. However, this comes at a cost of
approximately half our precision; better than LDA, but worse than KNN
and QDA. Applying this threshold to the out-of-folds sample data results
in the same number of false negatives as LDA, but with a higher amount
of uncertainty/variablitity, as seen from its higher standard deviation.
However, it predicts far less false positives than LDA (nearly 4 times
as less) with less uncertainty, but it trails KNN and QDA in every
metric, and in fact misclassifies nearly 4.5 times as many non-blue tarp
images as blue tarp compared to KNN.

#### $\underline{Random\space Forest}$

![](/images/rfTuning.png)

![](/images/haitiForest.png)

##### Random Forest Tuning Parameter Interpretation & Explanation

The only tuning parameter I selected for the random forest model was the
*mtry* parameter, which is the number of randomly sampled predictors
available as split candidates each time a split in a tree occurs. This
helps to decorrelate the number of trees used in the bagging process, as
it is less likely they will all use the strongest predictor as their top
split. By forcing each split in a tree to consider only a subset of the
predictors, the average of the resulting trees (i.e. the final model)
will be less variable. The chosen *mtry* value of 2 was selected using
Caret’s tuning functionality. I also ran 3 separate randomforest() BaseR
fits using different *mtry* values and plotted their test errors against
the number trees which also resulted in a suggested *mtry* value of 2.
Although *ntree* is a tuning parameter in BaseR randomforest(), it is
not available in Caret, and as my second plot shows, it is not as
crucial in the tuning process as performance plateaus after a certain
point; which appears to be around 500 trees, the default *ntree* value
of Caret’s “rf” method.

##### Random Forest ROC Curve

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-23-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-23-2.png" width="60%" style="display: block; margin: auto;" />

##### Random Forest Precision Recall Curve

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-24-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-24-2.png" width="60%" style="display: block; margin: auto;" />

##### Random Forest Confusion Matrix

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-25-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-25-2.png" width="60%" style="display: block; margin: auto;" />

##### Random Forest Threshold Rationale

Examining the mean Precision Recall Curve on the out-of-folds sample
data, it appears we can get very close to a sensitivity/recall of 1
(approximately 0.989) at a threshold of 0.005. In fact, this seems to be
our “threshold limit” as any lower values don’t improve our
sensitivity/recall, and in fact only lead to an increase in the false
positive rate. While its sensitivity given the chosen threshold slightly
lags all the prior models, all of its other metrics surpass the prior
models except for KNN (again, given their chosen thresholds), and in
fact, the average AUC of its Precision Recall Curve is the highest seen
thus far (something not dependent upon a threshold selection).

#### $\underline{SVM}$

![](/images/radialTuning.png)

##### SVM Tuning Parameter Interpretation & Explanation

The tuning parameters used for my SVM model were sigma and cost
(i.e. C). Although there are various different kernels for the SVM
method, I chose the radial kernel as it returned the highest accuracy
(given its optimal model), from the 3 different tuning kernels I ran;
the other 2 being linear and polynomial. Those two have been commented
out/excluded from the final output as their computational cost/runtime
was far too expensive/long. The first parameter in my model, sigma, is
the amount of curvature/flexibility allowed in the decision boundary,
and the second paramter, cost, is an error control measure; that is it
determines the number/severity of the violations to the margin (and thus
the hyperplane) that will be tolerated. It is essentially a budget for
how many *n* observations can violate the margin. Higher cost means we
are more tolerant of violations, and the margin widens, which amounts to
fitting the data “less hard” and obtaining a classifier that is
potentially more biased, but with lower variance. Essentially it
determines our bias-variance trade-off. These parameters were selected
using Caret’s tuning functionality. I passed in a series of possible
values for sigma & cost and allowed Caret to determine the parameters by
deciding which combination would return the highest accuracy; which in
this instance was a sigma of 2 and a cost of 100.

##### SVM ROC Curve

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-27-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-27-2.png" width="60%" style="display: block; margin: auto;" />

##### SVM Precision Recall Curve

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-28-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-28-2.png" width="60%" style="display: block; margin: auto;" />

##### SVM Confusion Matrix

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-29-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-29-2.png" width="60%" style="display: block; margin: auto;" />

##### SVM Threshold Rationale

Examining the mean Precision Recall Curve on the out-of-folds sample
data, it appears we can get to a sensitivity/recall of exactly 1 at a
threshold of 0.00369. Applying this threshold to the out-of-folds sample
data resulted in an accuracy of over 97%, but most importantly
accomplished my goal of providing aid to everyone! Not a single person
was overlooked (0 false negatives), and although the false positive rate
was higher than KNN, Random Forest & QDA, it was still significantly
lower than that of LDA & Logistic regression, and in my opinion, not
high enough to offset the fact that not a single displaced person was
overlooked.

#### **K-Folds Out of Sampling Performance Table**

| Method                   | KNN (*k*=3) | LDA       | QDA       | Logistic Regression | Random Forest(*mtry*=2) | SVM(sigma=2, cost=100) |
|--------------------------|-------------|-----------|-----------|---------------------|-------------------------|------------------------|
| Accuracy                 | 0.9922044   | 0.8786545 | 0.978827  | 0.9673314           | 0.9898325               | 0.9771667              |
| AUC                      | 0.9957232   | 0.9888613 | 0.9982067 | 0.9985065           | 0.994293                | 0.9997061              |
| ROC                      | See Above   | See Above | See Above | See Above           | See Above               | See Above              |
| PR_AUC                   | 0.9788371   | 0.8593759 | 0.9659138 | 0.9747459           | 0.9873107               | 0.9919693              |
| Prec_Rec Curve           | See Above   | See Above | See Above | See Above           | See Above               | See Above              |
| Threshold                | 0.062       | 1.88^{-4} | 0.021     | 0.0135              | 0.005                   | 0.00369                |
| Sensitivity=Recall=Power | 0.9925816   | 0.9901112 | 0.9920865 | 0.9901112           | 0.9891284               | 1                      |
| Specificity=1-FPR        | 0.9921919   | 0.8749733 | 0.978389  | 0.966579            | 0.9898561               | 0.9764126              |
| FDR                      | 0.191699    | 0.7925997 | 0.3967957 | 0.5048314           | 0.2358951               | 0.4142749              |
| Precision=PPV            | 0.808301    | 0.2074003 | 0.6032043 | 0.4951686           | 0.7641049               | 0.5857251              |

### **Hold-Out Sample Performance**

#### $\underline{Data\space Reading\space and\space Cleaning}$

#### $\underline{KNN}$

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-31-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-31-2.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-31-3.png" width="60%" style="display: block; margin: auto;" />

#### $\underline{LDA}$

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-32-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-32-2.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-32-3.png" width="60%" style="display: block; margin: auto;" />

#### $\underline{QDA}$

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-33-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-33-2.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-33-3.png" width="60%" style="display: block; margin: auto;" />

#### $\underline{Logistic\space Regression}$

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-34-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-34-2.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-34-3.png" width="60%" style="display: block; margin: auto;" />

#### $\underline{Random\space Forest}$

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-35-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-35-2.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-35-3.png" width="60%" style="display: block; margin: auto;" />

#### $\underline{SVM}$

<img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-36-1.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-36-2.png" width="60%" style="display: block; margin: auto;" /><img src="CourseProject_ReadMe_files/figure-gfm/unnamed-chunk-36-3.png" width="60%" style="display: block; margin: auto;" />

#### **Hold-Out Sample Performance Table**

| Method                   | KNN (*k*=3) | LDA       | QDA       | Logistic Regression | Random Forest(*mtry*=2) | SVM(sigma=2, cost=100) |
|--------------------------|-------------|-----------|-----------|---------------------|-------------------------|------------------------|
| Accuracy                 | 0.9895089   | 0.963382  | 0.9533933 | 0.995462            | 0.9715734               | 0.9133824              |
| AUC                      | 0.9444475   | 0.9921155 | 0.9915001 | 0.9994131           | 0.9804577               | 0.943833               |
| ROC                      | See Above   | See Above | See Above | See Above           | See Above               | See Above              |
| PR_AUC                   | 0.6913832   | 0.683106  | 0.76237   | 0.969352            | 0.723921                | 0.2802655              |
| Prec_Rec Curve           | See Above   | See Above | See Above | See Above           | See Above               | See Above              |
| Threshold                | 0.075       | 0.005     | 0.008     | 0.75                | 0.0095                  | 10^{-5}                |
| Sensitivity=Recall=Power | 0.8893646   | 0.9763812 | 0.9672652 | 0.9808011           | 0.9625                  | 0.7872928              |
| Specificity=1-FPR        | 0.9902377   | 0.9632874 | 0.9532924 | 0.9955687           | 0.9716394               | 0.9143                 |
| FDR                      | 0.601325    | 0.8378391 | 0.8690294 | 0.3830314           | 0.8019356               | 0.9373341              |
| Precision=PPV            | 0.398675    | 0.1621609 | 0.1309706 | 0.6169686           | 0.1980644               | 0.0626659              |

### **Conclusions**

#### 1. Discussion of Best Performing Algorithm(s) in the Cross-Validation and Hold-Out Data

When examining the K-Folds out-of-sample performance table at first
glance, it appears that on an individual basis each algorithm performed
incredibly well. In fact, besides LDA, for the ROC curves they all
returned average AUC values (across folds) over 0.99 (and LDA was
incredibly close with a value of 0.988) and for the Precision Recall
curves, they all returned average AUC values over 0.96 (except LDA
again, with an average PR AUC of 0.859). Yet going beyond these simple
point estimates, we see there is one algorithm that stood out above the
rest; the Support Vector Machine. Not only did it have the highest AUC
for both the ROC & Precision Recall Curves, but it also had the least
amount of uncertainty/variability in its estimates, as evident by its
lowest standard deviation for both AUC estimates. However, performance
evaluation became a bit murkier when it came time to set the threshold
for each method. Given my preference for a high sensitivity to limit the
number of overlooked displaced people, the threshold was set very low
for each algorithm. In doing so, SVM achieved a perfect sensitivity and
had 0 false negatives, however, this came at a fairly great cost of
precision, as it dropped to approximately 0.58, the third lowest of all
the methods. It is here that KNN stood out, as it had a near perfect
sensitivity (given my chosen threshold) of approximately 0.9926, and
also had the highest precision among all the algorithms of approximately
0.808301. Since we definitely do have a finite amount of manpower to
help provide aid to all the displaced people, we want to ensure we
correctly identify as many displaced people (with as much certainty) as
possible while simultaenously limiting our number of false positives, so
resources can be properly allocated to provide food and water to as many
displaced people as possible without sending it to too many locations
that don’t actually contain displaced individuals. Taking this into
account, KNN and Random Forest appear to best accomplish this goal, as
they both have sensitivities of approximately 0.99, and false discovery
rates around 0.20 (the lowest of all the methods). It should be noted
however, that when it comes to variability and uncertainty, LDA had the
lowest standard deviation across FDR, precision and sensitivity (except
for SVM on that last one since its theshold was set so low it had
perfect sensitivity across every fold).

Applying our models to the final hold-out data, we see some fairly
different results. While all the models still have high AUC’s for their
ROC curves, SVM now has the lowest value, as opposed to the best that it
had against the cross-validation data. The AUC of all the precision
recall curves dropped significantly too (except for logistic
regression), but again, SVM was hit the hardest, as it went from best to
worst in this metric as well. Although all of the models still have high
overall accuracies, many of them saw a slight decrease in this measure
and only two saw an increase; LDA and logistic regression. When we dive
deeper and look across all the metrics for the hold-out data there is
clearly one “winner” above the rest; logistic regression. Its ROC curve
AUC increased to a near perfect value of 1 (0.9994 to be more precise),
one of 2 models to see an increase in this metric; the other being LDA,
and it is the only model to maintain a precision recall curve AUC over
0.95; the second highest being QDA at 0.76. These performance
disparities become even more apparent once a threshold value is set for
each model, as SVM is now the worst performing in each statistic and
logistic regression is the best performing. In fact it was the only
model to have a false discovery rate under 0.5 (0.383 to be more exact),
while achieving the highest recall value as well. KNN still performed
moderately well, as it had the second highest overall accuracy on the
hold-out data and second lowest false discovery rate (again lagging
logistic regression). These findings make it apparent that there is no
“one size fits all” model, and a variety of factors, such as the data
they are trained on & their tuning parameters, can have a large effect
on their performance in the field.

#### 2. Discussion/Analysis Justifying Why Findings Above are Compatible or Reconcilable

It is clear from the two metric tables that the models do not perform
consistently across data sets. By examining the differences in these
tables, specifically the direction each of the performance metrics for
each model went, I believe there is one primary reason for these
performance disparities; the bias-variance trade-off inherit in each
algorithm. The only models that saw an increase in their accuracy and
ROC curve AUC were LDA and logistic regression, high-bias algorithms
that make more assumptions about the form of the target model, and are
thus less susceptible to being influenced by any possible “noise”
specific to their training data set during the fitting procedure. These
models also saw the least amount of reduction in their precision recall
curve AUC (another metric not influenced by threshold selection).
Consequently, KNN and SVM are high-variance models that fit more
flexibly/closely to their training data, and thus, were possibly overfit
to the training set. This could explain why they saw the largest drops
in their ROC and Precision Recall AUCs when run against the hold-out
data; they captured too much “noise” in the training data that did not
apply to the hold-out set. I believe this theory is further supported by
the performance differences seen in QDA and Random Forest as well. While
both algorithms are more flexible than LDA and logistic regression, they
are not as variable/flexible as KNN and SVM. QDA is very similar to LDA
except that it assumes a different covariance matrix for each class in
its function (making it slightly more flexible and less bias), and
although random forest is made up of decision trees (a high-variance
model), its aggregation of a multitude of them into a final fit reduces
its overall variance. The fact that these models saw less of a reduction
in their AUC’s compared to KNN and SVM, furthers my belief that the
primary reason for these performance differences across the data sets is
due to the bias-variance trade-off. To put it simply, the more flexible
models overfit to the training data, while the more bias models did not.

#### 3. Recommendation & Rationale Regarding Chosen Algorithm For Detection of Blue Tarps

If I had to choose just one of these algorithms for detection of blue
tarps going forward, I would select logistic regression. While the easy
rationale would be to just say it was the best performer on the hold-out
set, I believe it is the best selection given the nature of the data and
its purpose. Since these models were trained on high resolution
geo-referenced imagery, and the predictors were just composed of 3 pixel
colors, I believe there’s too much noise that could be (and apparently
was) captured during the model fitting process. There are so many blue
things beyond just tarps (vehicles, clothes, pools, not to mention these
pictures were taken near a port city), that I believe the more flexible
models, such as KNN and SVM, would capture too much of this information
from this specific training data. A more biased model, such as logistic
regression is less likely to include this “noise” in its fit, which
would make it more applicable for future blue tarp detection missions
(so long as they’re not in Santorini, Greece).

#### 4. Discussion of the Relevance of the Metrics Calculated in the Tables for this Context

Starting off, I believe AUC (of both ROC and Precision Recall curves) to
be one of the more important metrics regardless of context. They let us
know from the get-go if our model is better (or worse) than just
randomly guessing, and if further tuning, data gathering or an entirely
different approach should be taken towards the goal at hand. For this
specific context though, I believe the most important metric is
sensitivity/recall, followed closely by false discovery rate. As I
previously stated in Part 1 and further up in this submission, the most
important aspect to me of this mission is making sure as few displaced
people are overlooked/forgotten as possible, which is why all of my
thresholds were selected with a high sensitivity in mind; to minimize
the false negative rate. With that in mind, I believe the threshold
should be set low, but high enough that we aren’t just predicting blue
tarp every time, while at the same time keeping our false discovery rate
within a reasonable limit, as we do live in the real world and resources
are finite, so they must be allocated efficiently. Unfortunately, I did
not have the time to do the Cost-Benefit analysis to determine this
“reasonable” level, but believe false discovery rate should be related
to a budgetary limit as assessed by a governing body. As we saw with
Haiti, over \$3.5 billion in relief funding was received, and if the
budget allows for continual search/rescue parties to be sent out, then
they should be. I personally do not agree with some of my classmates’
belief that fruitless ventures could undermine morale; those rescue
workers volunteered for that job with the purpose of providing hope and
raising morale. The job IS searching for and finding people to provide
them relief. Finally, in this specific context, I believe specificity is
important due to the fact that the proportion of blue tarps/displaced
people compared to every other possible image classification is very
low, and as such the false positive rate should be low, meaning the
specificity should be high. If it were to ever get too low, than many
resources would go to waste (I personally feel this metric is most
important in medical diagnosis).

#### 5. Why Precision Recall over ROC Curves?

I believe for this specific assignment Precision Recall curves are much
more helpful than ROC curves due to the disproportion evident in the two
classes of Blue Tarp vs. No Tarp. There are significantly more No Tarp
observations than Blue Tarp observations, and we are not interested in
predicting/identifying locations where there is No Tarp/displaced
person. Precision Recall curves do not consider true negatives, whereas
ROC curves do, and thus give a more optimistic view of model performance
for this situation, since proportionally there are just WAYYYY more No
Tarp images. What I am most interested in is identifying/finding as many
blue tarps (the minority class in this classification problem) as
possible, while minimzing the number of false positives and negatives,
and a precision recall curve better assists me in accomplishing this.

#### 6. Suggested Improvement to Training Method/Data

This may be considered a “weak” conclusion at best, and I’m sure it
would only make the project much more complex/strenuous for future
cohorts if it were to be applied, but while running our Part 2 models,
specifically Random Forest, it got me thinking about the Microsoft
Kinect paper. I’m not sure if this could be captured from the
geo-referenced images or not, but could some sort of “point-to-point”
correspondences be trained on thousands of images of blue tarps to form
some sort of bounding or shape feature, as an additional predictor. I’m
not sure if this could be done using Random Forest on the provided
images, like it was with the kinect, due to perhaps some lack of depth
feature that the kinect provides that the images do not, but I remember
reading an article a while back on neural networks trying this (I
believe using the Keras package). If it is possible to add this as some
sort of “dimensional” predictor(s), maybe it could then be used when
training our models to help differentiate tarps from the tons of other
blue items, especially things like cars. Regardless, thanks for a great
semester Professor Schwartz!!
