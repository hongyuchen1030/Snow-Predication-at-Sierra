

W.I.P: 


SWE: Prediction: how much does sst constrain the swe std deviation at every point. run 10 000 esembles forward in time, and get their probability distribution. Use 12 months SST from last year's april to this year's april to predict SWE on April 1st this year. T2m has more time sensitive, use 2 years temp for daily values until April 1st, so as the tp. Try to predict the Jan 1 st snowpack, Jan 15th snowpack..... predict the snowpack from Jan to April and see how fast we lose the accuracy. Also use SWE on DEC 31th as an input. 


Question:
For the SWE data:
dimensions:
    time = 365
    Stats = 5
    Latitude = 4050
    Longitude = 5175

Do the 5 entries along the Stats axis correspond to:

ensemble members / realizations,
predefined summary statistics (e.g., mean, min, max),
or something else? Should be mean and quantiles, Stats[0] should be mean? Contact Alan for more details. 

Todo:
Try plot out the SWE data files, our job is figure out what each dimension means here, because 

  /global/cfs/projectdirs/m3522/datalake/UCLA_WUS_SNOWv1/WUS_UCLA_SR_v01_ALL_0_agg_16_WY2014_SD_SWE_SCA_POST.nc

  Using file, ncdump -h, ncdump -v Stats, and ncks -m, the relevant result is:

  dimensions:
      time = UNLIMITED ; // (365 currently)
      Longitude = 5175 ;
      Latitude = 4050 ;
      Stats = 5 ;

  variables:
      float SWE_Post(time, Stats, Latitude, Longitude) ;
      float SCA_Post(time, Stats, Latitude, Longitude) ;
      float SD_Post(time, Stats, Latitude, Longitude) ;
      int Stats(Longitude) ;
          Stats:units = "degrees_east" ;

  The key finding is that Stats is a dimension of length 5, but the file does not contain a valid coordinate variable for that dimension.
  Instead, it has a suspicious variable Stats(Longitude) with units="degrees_east", which is inconsistent with the Stats dimension and looks
  malformed.

  So from the file metadata alone, I can confirm:

  - Stats exists as a dimension with size 5.
  - SWE_Post, SCA_Post, and SD_Post all use that dimension.
  - There is no trustworthy label/meaning mapping for the 5 Stats entries in this file header.
  - The Stats variable itself appears corrupted or incorrectly defined.

  If you want, I can next inspect one of the chunked source files under chunked/2003/ to see whether the original per-tile files preserve a
  clean definition for Stats.


  Try to aggreate the SWE over the whole mountain range, do not try to predict grid level SWE, which is difficult. 