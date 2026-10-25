# Framing: Rawaj Pay mule detection

## Decision

For each instant transfer, the model should score the receiving account's mule risk within 300 ms. The five analysts review the top N cases each day, where N depends on their actual capacity, roughly a couple of hundred cases per day. Nothing should be blocked automatically. Missing a mule can cost the victim money and damage Rawaj Pay's reputation, while a false positive can freeze a legitimate customer's transfer and cause a complaint.

## Metric

I would use precision at k because the analysts can only review a limited number of cases each day. Precision and recall should also be measured at the chosen threshold, and expected cost should help compare thresholds. Accuracy is not enough because mule cases are rare, so a model can be highly accurate while missing many mules. The threshold should reflect analyst capacity and the real costs of false positives and false negatives, rather than using 0.5 by default.

## Baseline

The current baseline is Rawaj Pay's rules engine, which catches card fraud but does not adequately catch social-engineered transfers. The model should be compared with this existing process using the number of mule cases found, false-positive flags, analyst workload, and expected cost. The model should only be considered better if it improves detection without overwhelming the analysts or causing too many unnecessary customer disruptions.

## Classical ML or LLM

I would start with classical machine learning using structured transaction and account features, such as receiver age and recent incoming-transfer counts. It can produce risk scores quickly and consistently, which fits Rawaj Pay's high transaction volume and 300 ms latency requirement. Rules can provide a baseline, and a small LLM could optionally help explain flags in Arabic and English, but it should not make the main risk decision. Raw transaction data must not be sent to a foreign cloud.
