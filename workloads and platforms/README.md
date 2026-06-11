# Workloads

These workloads were created using traces from the LANL Mustang supercomputer. The traces span from 2011 to 2016. We selected three 7-day periods, each with a different job profile. We wanted periods with many small jobs (which allow more scheduling flexibility), many large jobs (which limit scheduling options), and a mix of both. We excluded unusual periods that did not reflect normal system behavior using the methodology below.

## Methodology

We analyzed 7-day windows throughout the entire trace period (2011-2016) using two criteria. First, we measured how regularly jobs arrived by analyzing submission patterns (Poisson similarity). Weeks where jobs arrived at steady intervals were preferred, while weeks with highly irregular patterns were excluded. Second, we classified weeks by their job size distribution, looking at the percentage of large jobs (≥120 nodes) and small jobs (≤10 nodes) to ensure we captured different types of workloads.

Three specific weeks were selected to represent different scenarios. The week of 2012-02-07 was chosen because it had a high proportion of large jobs (approximately 55% with ≥120 nodes), selected from weeks with at least 200 total jobs and regular arrival patterns. The week of 2015-08-06 represents the opposite extreme, with ~97% small jobs (≤10 nodes) and over 22,000 total submissions. The week of 2012-12-13 was selected to provide a balanced mix of small and large jobs. Each week was manually selected from the top candidates to ensure we covered all three workload types.

Each selected week was then converted to a Batsim-compatible format with 12 job profiles (4 CPU performance levels and 3 communication levels) assigned based on percentile ranking using a weighted combination (α_cpu=0.70, α_com=0.85).