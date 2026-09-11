# ACE2S Stochastic Ensemble Generation

## Project Goal

Our research project studies seasonal prediction of Western U.S. snow water equivalent (SWE). One part of the project requires a stochastic weather/climate emulator that can start from **one given atmospheric initial condition** and generate multiple plausible future atmospheric trajectories.

For this task, you will investigate **ACE2S**, the stochastic version of the Ai2 Climate Emulator. Here, “stochastic” means that the same initial atmospheric condition can generate multiple plausible future realizations:

$$
X_0 \xrightarrow{\epsilon_1} X_{1:T}^{(1)}, \qquad
X_0 \xrightarrow{\epsilon_2} X_{1:T}^{(2)}, \qquad
\ldots,\qquad
X_0 \xrightarrow{\epsilon_N} X_{1:T}^{(N)}.
$$

The goal for this stage is simple: **understand ACE2S and reproduce the released ACE2S inference run on Perlmutter.**

## Task 0 — Obtain Perlmutter/NERSC Access

Create a NERSC account and obtain access to **Perlmutter**.

Official instructions:

https://docs.nersc.gov/accounts/

Once your account is activated, learn how to SSH into Perlmutter and perform one successful SSH login test.

## Task 1 — Create the Project GitHub Repository

Create a **private GitHub repository** for this project and invite:

`hongyuchen1030`

as a collaborator.

The default `README.md` is sufficient.

After creating the repository, configure GitHub SSH access from your Perlmutter account and successfully clone the private repository onto Perlmutter.

## Task 2 — Understand ACE2S

Read the ACE2S paper and released documentation and determine how the model works and what is required to run it. 

Useful resources for your LLM models:

* **ACE2S paper:**
  https://arxiv.org/abs/2512.18224

* **ACE2S checkpoint and released files:**
  https://huggingface.co/allenai/HiRO-ACE

* **ACE code repository:**
  https://github.com/ai2cm/ace

* **ACE/FME documentation:**
  https://ai2-climate-emulator.readthedocs.io/en/stable/

The publicly released ACE2S model operates on a $1^\circ \times 1^\circ$ global grid, approximately 100 km resolution, and produces 6-hourly atmospheric simulations. It is stochastic, so multiple realizations can be generated from the same initial condition.

Your goal is to figure out:

* What initial condition ACE2S requires.
* What forcing/input data are required.
* What grid format and resolution it expects.
* What climate variables must be supplied.
* What variables ACE2S produces.
* How stochasticity is controlled during inference.

## Task 3 — Reproduce the Released ACE2S Run on Perlmutter

Use the **released ACE2S checkpoint, initial conditions, forcing data, and inference configuration provided by the authors** to reproduce one ACE2S inference run on Perlmutter.

The release contains the ACE2S checkpoint together with `initial_conditions/`, `forcing_data/`, and `ace2s_inference_config_global.yaml`. The published configuration points to `ACE2S.ckpt`, an initial-condition NetCDF file, forcing data, and writes 6-hourly ACE2S output.

Start with the authors' released data. Do not substitute our own project data at this stage.

The workflow should simply be:

```text
Released ACE2S checkpoint
        +
Released initial condition
        +
Released forcing data
        ↓
ACE2S inference configuration
        ↓
Run ACE2S on Perlmutter
        ↓
ACE2S atmospheric output
```

The official ACE2S inference step is:

```bash
python -m fme.ace.inference ace2s_inference_config_global.yaml
```

The released documentation treats this as a standalone ACE inference step.

If some required input is unclear or unavailable after checking the paper and released files, let us know. Once your NERSC account is ready, we can help point you to any additional data if needed.

### Running on Perlmutter

Before running ACE2S, read the NERSC documentation and understand the basic Perlmutter workflow:

* **Running jobs on Perlmutter:**
  https://docs.nersc.gov/systems/perlmutter/running-jobs/

* **General jobs documentation:**
  https://docs.nersc.gov/jobs/

* **Interactive jobs:**
  https://docs.nersc.gov/jobs/interactive/

* **Job best practices:**
  https://docs.nersc.gov/jobs/best-practices/

* **Coding agents on NERSC:**
  https://docs.nersc.gov/development/coding-agents/

Perlmutter uses Slurm for compute jobs, so understand the difference between an interactive allocation and a submitted batch job before running the model.

And always let us know if you have any questions for the run. 
## Week 1 Deliverable

Add a short Markdown report to your GitHub repository answering only the following questions:

| Question                                                                      | Finding |
| ----------------------------------------------------------------------------- | ------- |
| Were you able to successfully reproduce the released ACE2S run on Perlmutter? |         |
| What input data does ACE2S require?                                           |         |
| What grid format and resolution does ACE2S require?                           |         |
| What climate variables are required as input?                                 |         |
| What does the ACE2S output contain?                                           |         |
| What is the output grid/resolution?                                           |         |
| Were any required inputs or setup steps missing or unclear?                   |         |

The main goal for this stage is:

$$
\boxed{\text{Understand ACE2S} \rightarrow \text{Reproduce ACE2S on Perlmutter} \rightarrow \text{Document its input/output requirements}}
$$


# ACE-ERA5 Ensemble Generation — Week 2

As described in the ACE-ERA5 documentation, ACE-ERA5 has two types of models:

1. **Non-stochastic model:** each input entry generates one output realization. This is the original ACE-ERA5
2. **Stochastic model:** one input entry can generate multiple output realizations. This is the one you tried last week ACE2S, note, the ACE2S is just an academic product from a paper, adapted from the original ACE-ERA5

This week, we will try to use ACE-ERA5to synthesize the weather data needed for snow-data generation.

**Do not get intimidated by the data dump below. This week, we only aim to achieve ONE running pipeline test.** This means you only need to pick **one model** among the four models below and **one water year**.

For the one-water-year proof test, we can start with:

- **Model:** MIROC6
- **Member:** `r1i1p1f1`
- **Experiment:** `ssp370`
- **Water year:** WY2016

WY2016 means that the simulation starts in fall 2015 and runs through April 1, 2016.

We previously selected MIROC6 because it provides a clean, available, internally matched CMIP6–WUS-D3 pair for this proof-of-concept. We already have the corresponding CMIP6 predictor data and matching WUS-D3 d02 SWE simulation, so we can test the full pipeline without first solving a broader data-availability problem.

However, **feel free to pick a different model and a different year**. The goal this week is simply to demonstrate one complete working example of the lagged-ensemble data-inflation pipeline.



## The Data

We use WUS-D3 d02-generated SWE as our target. This SWE is the result of a physical weather simulation.

### Relationship between CMIP6 and WUS-D3 d02

Think of **CMIP6 → WUS-D3** as a two-stage physical climate simulation pipeline.

We use **four CMIP6 global climate models**. Each model produces its own simulation of Earth's climate on a global grid. From each CMIP6 simulation, we extract the 19 atmospheric, ocean, and land variables that we are interested in:

$$
\boxed{
\begin{aligned}
&\texttt{tos},\ \texttt{siconc},\ \texttt{rlut},\ \texttt{zg\_500},\ \texttt{ta\_850},\\
&\texttt{ua\_850},\ \texttt{va\_850},\ \texttt{ua\_200},\ \texttt{va\_200},\\
&\texttt{hus\_850},\ \texttt{psl},\ \texttt{tas},\\
&\texttt{zg\_50},\ \texttt{ta\_50},\ \texttt{ua\_50},\ \texttt{va\_50},\\
&\texttt{mrso},\ \texttt{thetao\_50m},\ \texttt{thetao\_100m}.
\end{aligned}}
$$

These variables are **global**: they describe climate conditions over the Earth rather than only over the Sierra Nevada.

Because we use four different CMIP6 models, for the same simulated period we effectively have:

$$
\text{CMIP model 1}\rightarrow X_1^{19},
\quad
\text{CMIP model 2}\rightarrow X_2^{19},
\quad
\text{CMIP model 3}\rightarrow X_3^{19},
\quad
\text{CMIP model 4}\rightarrow X_4^{19}.
$$

The four models therefore provide **four different physically generated realizations of the same 19 climate variables**.

Each of these CMIP6 simulations has a corresponding **WUS-D3 regional simulation**. WUS-D3 dynamically downscales the CMIP6 climate simulation over the western United States at much higher spatial resolution. For our project, we use the **WUS-D3 d02** domain.

The relationship can be visualized as:

```text
CMIP6 global climate model #1 ──→ WUS-D3 d02 #1 ──→ Sierra SWE #1
CMIP6 global climate model #2 ──→ WUS-D3 d02 #2 ──→ Sierra SWE #2
CMIP6 global climate model #3 ──→ WUS-D3 d02 #3 ──→ Sierra SWE #3
CMIP6 global climate model #4 ──→ WUS-D3 d02 #4 ──→ Sierra SWE #4

        GLOBAL CLIMATE                    REGIONAL DOWNSCALING
        19 variables                      high-resolution SWE
```

For our machine-learning dataset, the pairing is therefore essentially:

$$
\boxed{
\underbrace{\text{19 global CMIP6 variables}}_{\text{ML input}}
\quad\longrightarrow\quad
\underbrace{\text{corresponding WUS-D3 d02 Sierra SWE}}_{\text{ML target}}
}
$$

The important point is that **CMIP6 and WUS-D3 are not two unrelated datasets**. Each WUS-D3 simulation is associated with a particular parent CMIP6 simulation. Therefore, when constructing a training sample, we must preserve this parent relationship: CMIP model 1's 19-variable climate state is paired with the SWE produced by its corresponding WUS-D3 simulation, model 2 with model 2, and so on.

### Data Locations and CMIP6–WUS-D3 Pairings

#### CMIP6 19-Variable Global Inputs

The regridded 1.5° CMIP6 predictor fields are located at:

```text
/pscratch/sd/h/hyvchen/Snow-Predication-at-Sierra/cmip6_regridded_1p5deg/raw/
```

#### WUS-D3 d02 Daily SWE

The corresponding **daily WUS-D3 d02 SWE (`snow`)** files are located at:

```text
/global/cfs/projectdirs/m3522/datalake/WUS-D3/daily/<dataset_id>_{historical,ssp370}_bc/postprocess/d02/snow.daily.*.nc
```

Here, `<dataset_id>` identifies the WUS-D3 simulation associated with a particular CMIP6 parent model and ensemble member.

For example, for **MIROC6 (`r1i1p1f1`)**, the dataset ID is:

```text
miroc6_r1i1p1f1
```

Therefore, the corresponding WUS-D3 directories follow the pattern:

```text
/global/cfs/projectdirs/m3522/datalake/WUS-D3/daily/miroc6_r1i1p1f1_{historical,ssp370}_bc/postprocess/d02/
```

The daily SWE files inside these directories follow:

```text
snow.daily.*.nc
```

For example, a previously verified MIROC6 WUS-D3 d02 daily SWE file is:

```text
snow.daily.miroc6.r1i1p1f1.ssp370.bias-correct.d02.2015.nc
```

These are the **daily SWE fields**, not the processed April 1 SWE targets used by the CNN.

#### Parent Model Pairings

Each CMIP6 global simulation is paired with its corresponding WUS-D3 regional downscaling:

| CMIP6 Parent Model | CMIP6 Member | Corresponding WUS-D3 Dataset |
|---|---|---|
| EC-Earth3 | `r102i1p1f1` | `ec-earth3_r1i1p1f1_2_{historical,ssp370}*bc` |
| MIROC6 | `r1i1p1f1` | `miroc6_r1i1p1f1*{historical,ssp370}*bc` |
| MPI-ESM1-2-HR | `r3i1p1f1` | `mpi-esm1-2-hr_r3i1p1f1*{historical,ssp370}*bc` |
| TaiESM1 | `r1i1p1f1` | `taiesm1_r1i1p1f1*{historical,ssp370}_bc` |

The pairing must be preserved when constructing the training data: the global predictor fields from each CMIP6 parent simulation must be matched with the SWE generated by its corresponding WUS-D3 d02 regional simulation.

Conceptually:

```text
CMIP6 EC-Earth3       → WUS-D3 EC-Earth3       → Sierra SWE
CMIP6 MIROC6          → WUS-D3 MIROC6          → Sierra SWE
CMIP6 MPI-ESM1-2-HR   → WUS-D3 MPI-ESM1-2-HR   → Sierra SWE
CMIP6 TaiESM1         → WUS-D3 TaiESM1         → Sierra SWE
```

Thus, the CMIP6 global climate fields and WUS-D3 regional SWE are physically paired simulations rather than independent datasets.



## Non-Stochastic ACE-ERA5: Lagged Ensemble

We will eventually use an idea called a **lagged ensemble**. The purpose is to create multiple closely related weather realizations for what originally corresponds to one model and one water year.

Conceptually, we can think about shifting the November window:

```text
Nov 1–7  → ensemble member #1
Nov 2–8  → ensemble member #2
Nov 3–9  → ensemble member #3
...
```

Each shifted window represents one member of the eventual lagged-ensemble dataset.

**Important:** ACE-ERA5 itself requires a valid ACE-compatible atmospheric initial state. Do not directly use a simple 7-day average of the 19 CMIP6 variables as an ACE atmospheric initial state. Construct the corresponding valid ACE input/initialization required by the ACE-ERA5 implementation.

### Important: Only Generate ONE Ensemble Member This Week

**You do NOT need to construct the full November sliding-window ensemble.**

For this proof-of-concept, choose only:

- **ONE CMIP6/WUS-D3 parent model**
- **ONE water year**
- **ONE sliding-window / lagged member**
- **ONE ACE-ERA5 run**

For example:

```text
Model:        MIROC6
Member:       r1i1p1f1
Experiment:   ssp370
Water year:   WY2016
Window:       Nov 1–7, 2015
```

The goal is only to demonstrate one member of the future inflated dataset:

```text
ONE model
   ↓
ONE water year
   ↓
ONE selected lagged/sliding-window member
   ↓
ONE valid ACE-ERA5 initialization
   ↓
ONE non-stochastic ACE-ERA5 run
   ↓
ONE generated weather realization
   ↓
ONE precipitation trajectory
   +
ONE 2-m temperature trajectory
   ↓
Snow-17
   ↓
ONE corresponding SWE realization
```

In other words, **you are implementing only one member of what will eventually become the full lagged ensemble.** 

### Required SWE Output Grid

The final Snow-17 SWE field should be produced on the **WUS-D3 d02 spatial grid**, rather than retaining the native resolution of the weather-generation model.

Target grid:

```text
WUS-D3 d02
Native WRF Lambert grid
Approximately 9 km horizontal resolution
Grid size: 340 × 270
```

ACE-ERA5 and other weather-generation models may have different native horizontal resolutions. Therefore, appropriately map/regrid the required meteorological forcing to the WUS-D3 d02 grid before or as part of the Snow-17 calculation.

The desired output of this proof is:

```text
ONE ACE-ERA5 weather realization
        ↓
precipitation + 2-m temperature
        ↓
mapped to WUS-D3 d02 spatial grid
        ↓
Snow-17
        ↓
ONE SWE realization on the 340 × 270 WUS-D3 d02 grid
```

Do **not** simply return SWE at the native ACE-ERA5 resolution.

For later comparison with the existing CNN target, April 1 SWE can additionally be area-averaged over:

`35°N–42°N, 122.5°W–118°W`

but preserve the full WUS-D3 d02 SWE grid as the primary output of this experiment.