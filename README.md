# EPHIRD
# Evaluating Phosphorus Impacts of Residential Development (EPHIRD) is a model used for calculating the freshwater catchement phophorus pollution budgets associated with new developments 

# How to Run the App

**Step 1**: Install Python

If you do not have Python installed:
Download the latest version of Python from python.org  
Follow the installation instructions. Ensure that you check the option to 'Add Python to PATH' during installation


**Step 2**: Download the App

Visit the GitHub repository for the app (https://github.com/EPHIRDdev/EPHIRD)  
Click on the green "Code" button near the top of the page  
In the dropdown menu, click "Download ZIP"  
Once downloaded, extract the ZIP file to a folder on your computer  


**Step 3**: Navigate to the App Directory


Open your command prompt or terminal from the Windows search bar  
Right click on the folder containing the app  
Select Copy as path  
In the command prompt or terminal execute  

cd _PASTE YOUR PATH HERE_

_**or**_

Right click on the folder containing the app

Select Open in terminal

**Step 4**: Install pre-requisite Python packages

Execute the the following:

pip install Flask  
pip install pandas  
pip install numpy  
pip install matplotlib  

**Step 5**: Run the App

Execute the following:

flask run

_**or**_

python app.py 

**Step 6**: Access the App

You should now see the following 

                WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
                 * Running on http://127.0.0.1:5000
                Press CTRL+C to quit

ctrl + leftclick on "http://127.0.0.1:5000" 

_**or**_

Open a web browser  
Visit http://127.0.0.1:5000/ 



******The app should now be running in your browser******


**_Once the app has successfully launched, only steps 3,5,6 are necessary in order to re-launch_**



# User Guide 

Inputs:   
N = Number of dwellings  
A = Land area (ha)  
Oc = Average occupancy rate  
W = Average daily water usage per person (l)  
C = Wastewater nutrient concentration (mg/l)  
Ee = Existing nutrient export coefficient (kg/ha/yr)  
En = Additional nutrient export coefficient (kg/ha/yr)  
B = Nutrient budget (kg/yr)  

Please ensure that the field 'N' is filled before selecting the local authority as this value will be used, along with the local authority, to produce the estimate for the average occupancy rate  

You may select the dropdown and begin typing in order to find a Local Authority or WwTW

You may overwrite the Oc/C values given by the dropwdowns in order to use a custom value



**Datasets used are derived from:**

Ofwat (2023) Water company phosphorus discharge data 2011-2021. Available at: https://www.ofwat.gov.uk/wp-content/uploads/2023/06/Water-company-phosphorus-discharge-data-2011-2021.xlsx

ONS (2023) Household size. Available at: https://www.ons.gov.uk/datasets/TS017/editions/2021/versions/3



**Please cite this software as:**

Goulbourne, C. (2023) EPHIRD (Version 2.1.1) [Computer program]. Available at: https://github.com/EPHIRDdev/EPHIRD
