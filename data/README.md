# Dataset access and provenance

I used Kevin Hillstrom’s publicly shared 2008 email experiment. It contains 64,000 rows across three randomly assigned campaign arms. The analysis retains Mens E-Mail and No E-Mail, giving 42,613 rows. Campaign names describe merchandise, not customer gender.

- Original source and variable definitions: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
- Pinned scikit-uplift benchmark mirror: https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz
- SHA256: `bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f`

Run `python run.py download` from the repository root to retrieve and checksum-verify the exact dataset into `data/raw/hillstrom.csv.gz`. The source does not state an unrestricted redistribution licence, so raw rows and fitted models are not republished here. Local data and model outputs are excluded from Git. The complete aggregate data audit is in `reports/evaluation/data_audit.json`.

Seven pretreatment fields are used: recency, history, mens, womens, zip_code, newbie and channel. Assignment and all outcomes are excluded from targeting features. I normalised the source spelling Surburban to Suburban and retained identical-looking rows because no customer identifier proves duplication. The visit outcome covers the two weeks after the original campaign.
