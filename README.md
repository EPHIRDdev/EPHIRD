# EPHIRD
# Evaluating Phosphorus Impacts of Residential Development (EPHIRD) is a model used for calculating the freshwater catchement phophorus pollution budgets associated with new developments 

How to Run the App

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

cd PASTE YOUR PATH HERE

**or**

Right click on the folder containing the app

Select Open in terminal

**Step 4**

(This is only necessary the first time you run the app. You only need to complete this step once on the same computer)

Execute the the following in order to install the pre-requisite Python packages

pip install Flask  
pip install pandas  
pip install numpy  
pip install matplotlib  

**Step 5**: Run the App

Run the app by executing:

flask run

or

python app.py 

**Step 5**: Access the App

You should now see the following 

                WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
                 * Running on http://127.0.0.1:5000
                Press CTRL+C to quit

ctrl + leftclick on "http://127.0.0.1:5000" 

**or**

Open a web browser  
Visit http://127.0.0.1:5000/ 

The app should now be running in your browser.


Please cite this software as 
Goulbourne, C. (2023) EPHIRD (Version 2.1.1) [Computer program]. Available at https://github.com/EPHIRDdev/EPHIRD
