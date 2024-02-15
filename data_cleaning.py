# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:58:56 2024

@author: Nikola
"""

import pandas as pd 

df = pd.read_csv('D:/Python_e/ds_salary_proj/glassdoor_jobs.csv')







#salary parsing
df = df[df['Salary Estimate'] != '-1']

salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])

minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))

df['Hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)

df['Employer Provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

min_hr_emp = minus_Kd.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))

df['min_salary'] = min_hr_emp.apply(lambda x: int(x.split('-')[0]))

df['max_salary'] = min_hr_emp.apply(lambda x: int(x.split('-')[1]))

df['average salary'] = (df.min_salary + df.max_salary)/2

#Company name (remove rating)
df['company_name'] = df.apply(lambda x: x['Company Name'] if 'Rating' not in x or x['Rating'] < 0 else x['Company Name'][:-3], axis=1)

#Location (only state)
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])

df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

#Founded (change to age)
df['age'] = df['Founded'].apply(lambda x: x if x<0 else 2020 - x)

#job description (python, etc.)
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)

df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' or 'r-studio' in x.lower() else 0)

df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

df['excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

#dropping some columns

df_out = df.drop(['Unnamed: 0', 'min salary', 'max salary'], axis=1)

#exporting to csv file

df_out.to_csv("salary_cleaned.csv", index = False)

