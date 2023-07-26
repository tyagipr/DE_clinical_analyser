# DE_clinical_analyser

## code flow walkthrough

1. read data from patient, output event, chart event and lab event csvs
2. Filter all the data of one patient from output event file and calculate the cumulative urine output from it.
3. filter all the blood pressure and body temperature related data from chart event on the basis of patient id
4. filter all the lactose difference data from lab events on the basis of patient id

5. and combine all the data in result.csv file
   path - `result data is available in results/resuls.csv`
