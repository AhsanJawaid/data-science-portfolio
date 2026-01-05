import pandas as pd
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Configuration
NUM_RECORDS = 1000
ROLES = ["ICU Nurse", "ER Nurse", "Pediatric Nurse", "General Nurse", "Senior Nurse", "Junior Nurse"]
JOB_DESCRIPTIONS = [
    f"Required {random.choice(ROLES)}", 
    f"Hiring {random.choice(ROLES)}", 
    f"We are looking for {random.choice(ROLES)}", 
    f"Required Immediately: {random.choice(ROLES)}"
]
SKILLS_POOL = ["ICU", "Emergency Care", "Pediatrics", "Patient Care", "BLS/ACLS", "IV Therapy", "Telemetry", "Triage"]
REGIONS = {
    "North America": ["USA", "Canada"],
    "Europe": ["UK", "Germany", "France"],
    "Oceania": ["Australia", "New Zealand"],
    "Asia": ["India", "Philippines", "United Arab Emirates"]
}
EMP_STATUS = ["Full-time", "Part-time", "Contract", "PRN"]

data = []

for _ in range(NUM_RECORDS):
    # 1. Geographic Data
    region = random.choice(list(REGIONS.keys()))
    country = random.choice(REGIONS[region])
    
    # 2. Professional Details
    exp = random.randint(0, 25)
    role = random.choice(ROLES)
    jd = random.choice(JOB_DESCRIPTIONS)
    status = random.choices(EMP_STATUS, weights=[70, 15, 10, 5])[0]
    
    # 3. Salary Logic (Base + Experience Multiplier + Region Adjustment)
    base_salary = 50000
    reg_mult = 1.2 if region == "North America" else 1.0
    salary = int((base_salary + (exp * 2500)) * reg_mult * random.uniform(0.9, 1.1))
    
    # 4. Professional Resume Text
    skills = random.sample(SKILLS_POOL, k=random.randint(2, 4))
    name = fake.name()
    university = f"{fake.city()} Medical University"
    
    resume_text = (
        f"Candidate: {name}. Located in {country}. "
        f"Professional {role} with {exp} years of experience. "
        f"Specialized in {', '.join(skills)}. Graduate of {university}. "
        f"Seeking {status} opportunities."
    )
    
    # 5. Professional Selection Logic
    # Weighted score based on experience and role alignment
    score = 0
    if exp > 5: score += 2
    if "ICU" in role and "ICU" in skills: score += 3
    if "Senior" in role and exp > 10: score += 3
    
    selected = 1 if score >= 5 or (score >= 2 and random.random() > 0.7) else 0

    data.append({
        "full_name": name,
        "region": region,
        "country": country,
        "role": role,
        "experience_years": exp,
        "salary_expectation": salary,
        "employment_status": status,
        "resume_text": resume_text,
        "selected": selected,
        "job_description": jd
    })

# Create and Save DataFrame
df_pro = pd.DataFrame(data)
df_pro.to_csv("Professional_Nurse_Dataset.csv", index=False)

print(f"Generated {NUM_RECORDS} professional records.")
print(df_pro.head(3))