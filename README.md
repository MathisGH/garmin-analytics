## Anomaly Detection: Isolation Forest vs Autoencoder

Two unsupervised models were trained and compared: an Isolation Forest on
aggregated daily features, and a PyTorch autoencoder learning a compact
"health embedding" from raw multi-channel time series (HR, stress, body
battery, respiration).

Agreement between the two (Spearman correlation) started
strong (ρ=0.73) but dropped after extending the dataset (ρ≈0.30, not
significant). I knew that the small dataset would lead to bad results
but I still wanted to try to make it work in some way.

Rather than keep tuning against a metric that isn't reliably measurable at
this data scale, this line of work is considered complete for now, while still
fetching data everyday. Development continues on serving the existing models 
through a FastAPI service deployed on AWS Lambda.