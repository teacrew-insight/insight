import firebase_admin
from firebase_admin import credentials, firestore, storage
import requests
import time
from datetime import datetime
import pandas as pd
import math
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

global first_time

from google import genai
from google.genai import types
import json
import io
import pathlib
from typing import List, Dict, Optional

import os

# =======================(التعريف والمصادقة مع فير بيس)=====================

# اقرأ مسار ملف الخدمة من الانفايرومنت
firebase_credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH")

cred = credentials.Certificate(firebase_credentials_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
})
db = firestore.client()
bucket = storage.bucket()

# =======================(التعريف والمصادقة مع ديب سيك)=====================
api_url = "https://api.deepseek.com/v1/chat/completions"
api_key = os.getenv("DEEPSEEK_API_KEY")

# =======================(Google Gemini Authentication)=====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# =======================(تحميل نموذج الـ Embeddings)=====================
# Initialize the sentence transformer model for creating embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# =======================(الشروط والقواعد الذي يحتاج ان يتبعها الاي اي)=====================
SYSTEM_PROMPT = """

"""


# =====================================================================================(التعريف بالطلاب باستخدام اكسيل واضافتهم لفاير بيس)=====================
# ====================Main Function=================================
def process_excel_data(file_path):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name='Form Responses 1')

        # Select only the columns we need (A, B, C, D, and the Big 5 + learning styles columns)
        # Note: Column indices are 0-based in pandas
        required_columns = [
            'شعبة', 'الأسم', 'ماهو رقمك الأكاديمي', 'معدلك التراكمي (من 5)',
            'EXTRAVERSION الانبساطية', 'AGREEABLENESS القبول',
            'Conscientousness الضمير الحي', 'Neuroticism الاستقرار العاطفي',
            'Opennss to Experience الانفتاح على الخبرة',
            '1. Inferencng', '2. Note-Taking', '3. Dictionary use',
            '4.Oral Repeteition', '5. Activation',
            '6. Visual and Auditory Encoding', '7. Structural Encoding',
            '8. Self-Inflation', '9. Selective Attention'
        ]

        # Filter the dataframe to only include required columns
        df_filtered = df[required_columns]

        # Rename columns to English for easier processing
        column_mapping = {
            'شعبة': 'clas_name',
            'الأسم': 'name',
            'ماهو رقمك الأكاديمي': 'id',
            'معدلك التراكمي (من 5)': 'gpa',
            'EXTRAVERSION الانبساطية': 'Big1',
            'AGREEABLENESS القبول': 'Big2',
            'Conscientousness الضمير الحي': 'Big3',
            'Neuroticism الاستقرار العاطفي': 'Big4',
            'Opennss to Experience الانفتاح على الخبرة': 'Big5',
            '1. Inferencng': 'inferencing',
            '2. Note-Taking': 'note_taking',
            '3. Dictionary use': 'dictionary_use',
            '4.Oral Repeteition': 'oral_repetition',
            '5. Activation': 'activation',
            '6. Visual and Auditory Encoding': 'visual_auditory_encoding',
            '7. Structural Encoding': 'structural_encoding',
            '8. Self-Inflation': 'self_inflation',
            '9. Selective Attention': 'selective_attention'
        }

        df_filtered = df_filtered.rename(columns=column_mapping)

        # Define learning style names
        lsnames = [
            'inferencing',
            'note_taking',
            'dictionary_use',
            'oral_repetition',
            'activation',
            'visual_auditory_encoding',
            'structural_encoding',
            'self_inflation',
            'selective_attention'
        ]

        # Convert to list of dictionaries and add age and learning style results
        data = []
        for record in df_filtered.to_dict('records'):
            # Get learning style results in order
            lsresult = [record[ls] for ls in lsnames]

            # Create new record with all fields
            new_record = {
                'clas_name': "Class " + record['clas_name'],
                'name': record['name'],
                'id': record['id'],
                'GPAout5': record['gpa'],
                'GPSout100': record['gpa'] / 5,
                'age': 24,  # Fixed age for all students
                'Big1': record['Big1'] / 5,
                'Big2': record['Big2'] / 5,
                'Big3': record['Big3'] / 5,
                'Big4': record['Big4'] / 5,
                'Big5': record['Big5'] / 5,
                'LSnames': lsnames,  # List of learning style names
                'LSResult': lsresult  # List of learning style values
            }
            data.append(new_record)
        students = assign_traits(data)
        i = 0
        for student in students:
            if i > 0 and i % 10 == 0:
                print(f"sleep number {i / 10}")
                time.sleep(10)  # 5 second pause
            i = i + 1;
            student['strength'], student['weak'] = generate_feedback(student)
            print(student['name'])
            print("has been completed")
            time.sleep(3)  # 5 second pause
        add_students_to_firebase(students)
    except Exception as e:
        print(f"Error identifying student {str(e)}")
        students = []
    return students


# ===========================supportive functions=============================
# ====================trait assignment=============================
def assign_traits(data):
    try:
        for student in data:
            """Assign 3 traits based on Big5 and learning style scores"""
            traits = []
            TRAIT_POOL = {
                "Creative": {"Big5": ["Big5"], "learning_styles": ["visual_auditory_encoding", "structural_encoding"]},
                "Analytical": {"Big5": ["Big3"], "learning_styles": ["inferencing", "note_taking"]},
                "Sociable": {"Big5": ["Big1"], "learning_styles": ["oral_repetition", "activation"]},
                "Organized": {"Big5": ["Big3"], "learning_styles": ["note_taking", "selective_attention"]},
                "Adaptable": {"Big5": ["Big5"], "learning_styles": ["activation", "dictionary_use"]},
                "Perfectionist": {"Big5": ["Big3", "Big4"], "learning_styles": ["structural_encoding"]},
                "Introspective": {"Big5": ["Big4"], "learning_styles": ["self_inflation"]},
                "Energetic": {"Big5": ["Big1"], "learning_styles": ["activation"]},
                "Detail-oriented": {"Big5": ["Big3"], "learning_styles": ["note_taking", "structural_encoding"]},
                "Independent": {"Big5": ["Big5"], "learning_styles": ["inferencing", "self_inflation"]},
                "Collaborative": {"Big5": ["Big2"], "learning_styles": ["oral_repetition"]},
                "Innovative": {"Big5": ["Big5"], "learning_styles": ["visual_auditory_encoding"]}
            }
            # Get top 2 Big5 traits (highest scores)
            big5_scores = {f"Big{i + 1}": student[f"Big{i + 1}"] for i in range(5)}
            top_big5 = sorted(big5_scores.items(), key=lambda x: x[1], reverse=True)[:2]

            # Get top 2 learning styles (highest scores)
            ls_scores = {ls: student['LSResult'][i] for i, ls in enumerate(student['LSnames'])}
            top_ls = sorted(ls_scores.items(), key=lambda x: x[1], reverse=True)[:2]

            # Find matching traits
            for trait, criteria in TRAIT_POOL.items():
                big5_match = any(b[0] in criteria["Big5"] for b in top_big5)
                ls_match = any(ls[0] in criteria["learning_styles"] for ls in top_ls)

                if big5_match and ls_match and len(traits) < 3:
                    traits.append(trait)

            # If not enough traits matched, fill with random from top matches
            while len(traits) < 3:
                available = [t for t in TRAIT_POOL.keys() if t not in traits]
                if not available:
                    break
                traits.append(available[0])
            student['skills'] = traits
    except Exception as e:
        print(f"Error traiting student {str(e)}")
    return data


# =====================strength and weaknesess assignment============================
def generate_feedback(student):
    """Generates feedback using GPA, Big5, and learning styles"""
    prompt = f"""
    Analyze this student profile:
    - GPA: {student['GPAout5']}/5
    - Big Five Traits:
      * Extraversion: {student['Big1']}
      * Agreeableness: {student['Big2']}
      * Conscientiousness: {student['Big3']}
      * Neuroticism: {student['Big4']}
      * Openness: {student['Big5']}
    - Top Learning Styles: 
      {', '.join(
        f"{style}: {score}"
        for style, score in sorted(
            zip(student['LSnames'], student['LSResult']),
            key=lambda x: x[1],
            reverse=True
        )[:3]
    )}

    Generate:
    3 POSITIVE academic/work strengths and 3 CONSTRUCTIVE areas for growth.
    Use neutral, encouraging language, and keep it short. NO HEADERS, Example:
    1. First strength
    2. Second strength
    3. Third strength

    1. First weakness
    2. Second weakness
    3. Third weakness
    """
    try:
        response = chat_with_deepseek(prompt)

        parts = response.split("\n\n")
        strengths = parts[0].strip() if len(parts) > 0 else ""
        weaknesses = parts[1].strip() if len(parts) > 1 else ""
        return strengths, weaknesses

    except Exception as e:
        print(f"Error generating strength and weaknesess student {str(e)}")
        return None


# =============================additing students to firebase=====================
def add_students_to_firebase(students):
    for student in students:
        try:
            # Add student to Firebase in their section collection
            doc_ref = db.collection("users").document(str(student['id']))

            doc_ref.set(student)
        except Exception as e:
            print(f"Error adding student {student.get('name')} to Firebase: {str(e)}")


# ========================================================================================(حساب المتوسط الحسابي لكل شعبة ورفعها في فايربيس)=====================
# ====================Main Function=================================
def analyze_class_students():
    students = readData("users")
    # Big Five trait mapping
    big5_mapping = {
        'Big1': 'Extraversion',
        'Big2': 'Agreeableness',
        'Big3': 'Conscientiousness',
        'Big4': 'Neuroticism',
        'Big5': 'Openness'
    }

    # Learning style categories mapping
    learning_style_categories = {
        'visual_reading': ['inferencing', 'note_taking', 'dictionary_use', 'structural_encoding'],
        'auditory': ['oral_repetition', 'visual_auditory_encoding'],
        'kinesthetic': ['activation', 'self_inflation', 'selective_attention']
    }
    # Required fields for each student
    required_fields = {
        'class': 'clas_name',
        'big5': ['Big1', 'Big2', 'Big3', 'Big4', 'Big5'],
        'learning_styles': ['LSnames', 'LSResult']
    }
    # Organize students by class
    classes = {}
    for student in students:
        # Skip student if any required field is missing
        skip_student = False

        # Check class name
        if required_fields['class'] not in student:
            skip_student = True
        if required_fields['class'] in student:
            x = "x"
            if type(student['clas_name']) != type(x):
                skip_student = True

        # Check Big Five traits
        for trait in required_fields['big5']:
            if trait not in student:
                skip_student = True
                break

        # Check learning style data
        for ls_field in required_fields['learning_styles']:
            if ls_field not in student:
                skip_student = True
                break

        if skip_student:
            continue
        class_name = student['clas_name']
        if class_name not in classes:
            classes[class_name] = {
                'students': [],
                'big5_totals': {'Big1': 0, 'Big2': 0, 'Big3': 0, 'Big4': 0, 'Big5': 0},
                'ls_counts': {'visual_reading': 0, 'auditory': 0, 'kinesthetic': 0},
                'ls_scores': {'visual_reading': 0, 'auditory': 0, 'kinesthetic': 0},
                'all_ls_names': student['LSnames'],  # Store learning style names
                'all_ls_sums': [0] * len(student['LSResult']),  # Initialize sums for each LS
                'ls_counts_per_style': [0] * len(student['LSResult'])  # Count for averaging
            }
        classes[class_name]['students'].append(student)

    # Calculate averages for each class
    for class_name, class_data in classes.items():
        num_students = len(class_data['students'])

        # Calculate Big5 averages
        for student in class_data['students']:
            for trait in ['Big1', 'Big2', 'Big3', 'Big4', 'Big5']:
                class_data['big5_totals'][trait] += student[trait]

        # Normalize Big5 totals to get averages
        for trait in class_data['big5_totals']:
            class_data['big5_totals'][trait] /= num_students

        # Calculate learning style preferences
        for student in class_data['students']:
            # Match each student's learning styles to categories
            for i, (ls_name, ls_score) in enumerate(zip(student['LSnames'], student['LSResult'])):
                class_data['all_ls_sums'][i] += ls_score
                class_data['ls_counts_per_style'][i] += 1

                for category, ls_list in learning_style_categories.items():
                    if ls_name in ls_list:
                        class_data['ls_counts'][category] += 1
                        class_data['ls_scores'][category] += ls_score

        # Calculate percentages for learning style categories
        total_ls = sum(class_data['ls_counts'].values())
        for category in class_data['ls_counts']:
            if total_ls > 0:
                class_data['ls_counts'][category] = (class_data['ls_counts'][category] / total_ls) * 100
                class_data['ls_scores'][category] /= (class_data['ls_counts'][category] * total_ls / 100) if \
                    class_data['ls_counts'][category] > 0 else 1

        # Calculate averages for all learning styles
        class_data['all_ls_avgs'] = [
            sum_val / count if count > 0 else 0
            for sum_val, count in zip(class_data['all_ls_sums'], class_data['ls_counts_per_style'])
        ]

    # Generate output in the requested format
    results = []
    for class_name, class_data in classes.items():
        # Prepare learning styles output
        ls_percentages = class_data['ls_counts']
        dmls = ""

        # Format learning styles percentages
        visual_reading = round(ls_percentages['visual_reading'])
        auditory = round(ls_percentages['auditory'])
        kinesthetic = round(ls_percentages['kinesthetic'])
        dmls = f"Visual-Reading learners ({visual_reading}%)\nAuditory learners ({auditory}%)\nKinesthetic learners ({kinesthetic}%)"

        # Prepare Big5 output
        big5_avgs = class_data['big5_totals']
        dpt = ""

        # Categorize each trait as High/Moderate/Low based on thresholds
        for trait, value in big5_avgs.items():
            trait_name = big5_mapping[trait]
            if value >= 0.75:
                dpt = dpt + (f" High {trait_name.lower()}")
            elif value >= 0.50:
                dpt = dpt + (f" Moderate {trait_name.lower()}")
            elif value >= 0.50:
                dpt = dpt + (f" Medium {trait_name.lower()}")
            else:
                dpt = dpt + (f" Low {trait_name.lower()}")
            dpt = dpt + "\n"

        # Create the output dictionary
        class_result = {
            'DMLS': dmls,
            'DPT': dpt,
            'ABF1': round(class_data['big5_totals']['Big1'], 1),
            'ABF2': round(class_data['big5_totals']['Big2'], 1),
            'ABF3': round(class_data['big5_totals']['Big3'], 1),
            'ABF4': round(class_data['big5_totals']['Big4'], 1),
            'ABF5': round(class_data['big5_totals']['Big5'], 1),
            'ALSnames': class_data['all_ls_names'],
            'ALSresult': [round(score, 1) for score in class_data['all_ls_avgs']],
            'class_name': class_name,
            'ClassName': class_name
        }

        results.append(class_result)
    add_average_to_firebase(results)


# ===========================supportive functions=============================
# ==============================adding the average of sections=======================
def add_average_to_firebase(classes):
    for clas in classes:
        try:
            # Add student to Firebase in their section collection
            class_number = clas['class_name'].split()[-1]  # Gets the last word ("1" from "Class 1")
            doc_ref = db.collection("Class1").document(str(class_number))

            doc_ref.set(clas)
            print(f"class {clas['class_name']} is added to firebase")
        except Exception as e:
            print(f"Error adding student {clas.get('class_name')} to Firebase: {str(e)}")


# =========================================================================================================(انشاء اشعارات  ورفعها في فايربيس)=====================
# ====================Main Function=================================
def generate_combined_averages_prompt():
    classes_data = readData('Class1')
    students = readData('users')
    # Initialize dictionaries to store totals and counts
    big5_totals = {'Big1': 0, 'Big2': 0, 'Big3': 0, 'Big4': 0, 'Big5': 0}
    ls_totals = {}
    ls_counts = {}
    total_sections = len(classes_data)

    sum = 0
    i = 0
    for student in students:
        if 'GPAout5' not in student:
            continue
        else:
            if not (student['GPAout5'] > 0):
                continue
        sum = sum + student['GPAout5']
        i = i + 1
    avgpa = sum / i

    # First pass to initialize learning style totals based on first section's structure
    if classes_data:
        first_class = next(iter(classes_data.values())) if isinstance(classes_data, dict) else classes_data[0]
        ls_names = first_class['ALSnames']
        ls_totals = {name: 0 for name in ls_names}
        ls_counts = {name: 0 for name in ls_names}

    # Calculate totals across all sections
    for class_data in classes_data.values() if isinstance(classes_data, dict) else classes_data:

        # Sum Big5 scores
        for trait in ['Big1', 'Big2', 'Big3', 'Big4', 'Big5']:
            big5_totals[trait] += class_data[f'ABF{trait[-1]}']  # Using ABF1, ABF2 etc.

        # Sum learning style scores
        for name, score in zip(class_data['ALSnames'], class_data['ALSresult']):
            ls_totals[name] += score
            ls_counts[name] += 1

    # Calculate averages
    big5_avgs = {
        'Extraversion': big5_totals['Big1'] / total_sections,
        'Agreeableness': big5_totals['Big2'] / total_sections,
        'Conscientiousness': big5_totals['Big3'] / total_sections,
        'Neuroticism': big5_totals['Big4'] / total_sections,
        'Openness': big5_totals['Big5'] / total_sections
    }

    ls_avgs = {
        name: ls_totals[name] / ls_counts[name] if ls_counts[name] > 0 else 0
        for name in ls_totals
    }

    # Format the prompt
    prompt = f"""
    Based on the following average metrics about my students, generate 3 specific, actionable daily tips 
    to help me be a more effective teacher. Each tip should have:
    - A creative header (max 8 words)
    - A detailed body (max 20 words) with practical advice



    Student Data:
    - Personality (Big Five, 0-1 scale):
      Openness: {(big5_avgs['Openness']):.1f}, 
      Conscientiousness: {(big5_avgs['Conscientiousness']):.1f},
      Extraversion: {(big5_avgs['Extraversion']):.1f},
      Agreeableness: {(big5_avgs['Agreeableness']):.1f},
      Neuroticism: {(big5_avgs['Neuroticism']):.1f}

    Learning Styles (out of 5):
    """
    for name, avg in ls_avgs.items():
        prompt += f"- {name.replace('_', ' ').title()}: {avg:.1f}/5\n"
    prompt += f"""
    - Average GPA: {avgpa:.2f}/5.0

    Tips should consider:
    1. How personality traits affect classroom dynamics
    2. How to accommodate dominant learning styles
    3. How GPA might indicate needed adjustments
    4. Practical strategies that can be implemented today
    5. Balancing different student needs6. Make the tips specific, evidence-based, and immediately useful.

7. Do not include numbers in either the header nor the body

8. DO NOT: ask for missing info; if something isn’t in the instructional content, skip it.

9. DO NOT Use Special Characters for formatting.



The answer should have the header and the body ONLY, using the following format:

       <*Header 1*>

        <Body 1>



       <*Header 2*>

        <Body 2>



       <*Header 3*>

        <Body 3>
    """
    print(prompt)
    try:
        response = chat_with_deepseek(prompt)
        blocks = response.strip().split('\n\n')

        notifications = []
        notification_id = 1

        for block in blocks:
            # Split each block into lines and remove empty lines
            lines = [line.strip() for line in block.split('\n') if line.strip()]

            if len(lines) >= 2:
                header = lines[0].strip('<> ')  # Remove < > and whitespace
                header = header.strip('****')  # Remove < > and whitespace
                body = lines[1].strip('<> ')  # Remove < > and whitespace

                notifications.append({
                    'notification_id': notification_id,
                    'header': header,
                    'body': body
                })

                notification_id += 1
        add_notifications_to_firebase(notifications)

    except Exception as e:
        print(f"Error generating strength and weaknesess student {str(e)}")
    # Learning Styles section

    return None


# ===========================supportive functions=============================
# ==============================adding the notifications=======================
def add_notifications_to_firebase(notifications):
    for noti in notifications:
        try:
            # Add student to Firebase in their section collection
            doc_ref = db.collection("Notifications").document(str(noti['notification_id']))

            doc_ref.set(noti)
            print(f"class {noti['notification_id']} is added to firebase")
        except Exception as e:
            print(f"Error adding student {noti.get('notification_id')} to Firebase: {str(e)}")


# ========================================================================================(انشاء المجموعات للطلاب حسب الشعبة وترتيبها تلقائيا)=====================
# ====================Main Function=================================
# =======================(ارسال طلب تجميع قروبات لديب سيك واخذ اجابة منه)=====================
def group_students(groupNumber, GroupType, classgroup, descr):
    # Filter students by class
    records = readData("users")
    # Required fields for each student
    students = []
    for record in records:
        student = {
            'clas_name': record['clas_name'],
            'name': record['name'],
            'gpa': record['GPAout5'],
            'Extraversion': record['Big1'] * 5,
            'Agreeableness': record['Big2'] * 5,
            'Conscientiousness': record['Big3'] * 5,
            'Neuroticism': record['Big4'] * 5,
            'Openness': record['Big5'] * 5,
            'inferencing': record['LSResult'][0],
            'note_taking': record['LSResult'][1],
            'dictionary_use': record['LSResult'][2],
            'oral_repetition': record['LSResult'][3],
            'activation': record['LSResult'][4],
            'visual_auditory_encoding': record['LSResult'][5],
            'structural_encoding': record['LSResult'][6],
            'self_inflation': record['LSResult'][7],
            'selective_attention': record['LSResult'][8]
        }
        students.append(student)
    class_students = [student for student in students if student['clas_name'] == classgroup]

    if not class_students:
        return {"Notes": "No students found in this class", "GroupMembers": {}}
    print(len(class_students))
    # Calculate number of groups needed
    total_students = len(class_students)
    prompt = calculate_groups(total_students, groupNumber)

    # Create strict prompt for DeepSeek API
    prompt = prompt + f"""
    Grouping should be based on: {GroupType}.
    Additional instructions: {descr}.

    Important rules:
    1. Do not duplicate any student across multiple groups.
    2. Consider the following student attributes for grouping:
       - Big Five personality scores (out of 5, where 5 is max)
       - Learning style scores (9 types)(out of 5, where 5 is max)
       - GPA (out of 5)

    Students data:
    {class_students}

    Return the groups in the format:
    Group 1: name1, name2, ...
    Group 2: name3, name4, ...
    ...
    also add a note of how you did the grouping in short sentence and make it between <> 
    DO NOT include any text other than the groups and the note of the grouping reason
    """
    print(prompt)
    # Call the API (simulated in this example)
    try:
        result = chat_with_deepseek(prompt)
        # Split the text into two parts using '<' as the delimiter
        parts = result.split('<', 1)  # The '1' ensures we only split at the first occurrence

        # The first part is everything before '<'
        groupMembers = parts[0].strip()  # strip() removes any leading/trailing whitespace

        # The second part is everything after '<' (and we add the '<' back)
        Notes = '<' + parts[1].strip() if len(parts) > 1 else ''
        return groupMembers, Notes
    except Exception as e:
        print(e)
        groupMembers, Notes = "", ""
        return groupMembers, Notes


# ====================Supportive Functions=================================
# =======================(حساب العدد المثالي للمجموعات)=====================
def calculate_groups(total_students, groupNumber):
    t = 1
    while t:
        t = 0
        num_groups = math.floor(total_students / groupNumber)
        remainder = total_students % groupNumber
        if remainder == 0:
            adjusted_groupNumber = 0
            adjusted_num_groups = 0
            prompt = f"Group the following students into {num_groups} groups with {groupNumber} students per group"

        elif remainder <= groupNumber / 2:
            adjusted_groupNumber = groupNumber + 1
            adjusted_num_groups = remainder
            num_groups = num_groups - adjusted_num_groups
            prompt = f"Group the following students into {num_groups + adjusted_num_groups} groups: {num_groups} groups of {groupNumber} students and {adjusted_num_groups} groups of {adjusted_groupNumber} students"
            if num_groups < 0:
                groupNumber = groupNumber - num_groups
                t = 1

        else:
            adjusted_groupNumber = groupNumber - 1
            adjusted_num_groups = groupNumber - remainder
            num_groups = num_groups - adjusted_num_groups + 1
            prompt = f"Group the following students into {num_groups + adjusted_num_groups} groups: {num_groups} groups of {groupNumber} students and {adjusted_num_groups} groups of {adjusted_groupNumber} students"

    return prompt


# =======================(التحقق التلقائي من وجود مجموعة جديدة)=====================
def group_update_check():
    groups = readData("Groups")
    for group in groups:
        # Check if 'GroupMembers' key is missing (not just empty/None)
        if group.get('GroupMembers') is None:  # or 'GroupMembers' not in group
            try:
                groupNumber = group['groupNumber']
                GroupType = group['GroupType']
                classgroup = group['classgroup']
                descr = group['descr']

                # Generate new members and notes
                new_members, new_notes = group_students(groupNumber, GroupType, classgroup, descr)

                try:
                    # Update only the missing fields in Firestore
                    doc_ref = db.collection("Groups").document(str(group['id']))
                    doc_ref.update({
                        'GroupMembers': new_members,
                        'Notes': new_notes
                    })
                    print(f"Updated Group {group['id']} with new members.")
                except Exception as e:
                    print(f"Error updating group {group['id']}: {str(e)}")
            except Exception as e:
                print(f"Error updating group {group['id']}: {str(e)}")
        else:
            print(f"Group {group.get('id')} already has members. Skipping.")


# ========================================================================================(الجواب على اسئلة المعلم والتحقق من وجودها تلقائيا)=====================
# ====================Main Function=================================
class StudentRAG:
    def __init__(self):
        self.students_data = []
        self.student_embeddings = []
        self.student_texts = []

    def load_student_data(self):
        """Load and process student data from Firebase"""
        self.students_data = readData('users')
        self.prepare_embeddings()

    def create_student_text(self, student):
        """Create searchable text representation of student data"""
        # Extract learning style names and values
        ls_info = ""
        if 'LSResult' in student and 'LSnames' in student:
            ls_pairs = list(zip(student['LSnames'], student['LSResult']))
            ls_info = " ".join([f"{name}: {value}" for name, value in ls_pairs])

        # Create comprehensive text representation
        text = f"""
        Student Name: {student.get('name', 'Unknown')}
        Student ID: {student.get('id', 'Unknown')}
        Age: {student.get('age', 'Unknown')}
        Class: {student.get('clas_name', 'Unknown')}
        GPA (out of 5): {student.get('GPAout5', 'Unknown')}
        GPA (out of 100): {student.get('GPAout100', 'Unknown')}
        Skills: {', '.join(student.get('skills', []))}
        Strengths: {student.get('strength', 'Not specified')}
        Weaknesses: {student.get('weak', 'Not specified')}
        Big Five Personality Traits:
        - Openness: {student.get('Big1', 'Unknown')}
        - Agreeableness: {student.get('Big2', 'Unknown')}
        - Conscientiousness: {student.get('Big3', 'Unknown')}
        - Neuroticism: {student.get('Big4', 'Unknown')}
        - Extraversion: {student.get('Big5', 'Unknown')}
        Learning Styles: {ls_info}
        """
        return text.strip()

    def prepare_embeddings(self):
        """Create embeddings for all students"""
        self.student_texts = []
        for student in self.students_data:
            text = self.create_student_text(student)
            self.student_texts.append(text)

        # Create embeddings for all student texts
        if self.student_texts:
            self.student_embeddings = embedding_model.encode(self.student_texts)
        else:
            self.student_embeddings = np.array([])

    def find_relevant_students(self, query, top_k=3):
        """Find most relevant students based on the query."""
        # No embeddings available
        if not isinstance(self.student_embeddings, np.ndarray) or self.student_embeddings.size == 0:
            return []

        query_embedding = embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.student_embeddings)[0]

        # Sort by similarity and filter with threshold
        ranked = [(i, float(sim)) for i, sim in enumerate(similarities)]
        ranked.sort(key=lambda x: x[1], reverse=True)

        relevant_students = []
        for idx, sim in ranked[:top_k]:
            if sim >= 0.45:
                relevant_students.append({
                    'student_data': self.students_data[idx],
                    'student_text': self.student_texts[idx],
                    'similarity': sim
                })

        return relevant_students

    def search_by_id_or_name(self, identifier):
        """Search for a specific student by ID or name"""
        identifier = identifier.strip().lower()

        for i, student in enumerate(self.students_data):
            # Check ID match
            if str(student.get('id', '')).lower() == identifier:
                return [{
                    'student_data': student,
                    'student_text': self.student_texts[i],
                    'similarity': 1.0
                }]

            # Check name match (partial or full)
            student_name = str(student.get('name', '')).lower()
            if identifier in student_name or student_name in identifier:
                return [{
                    'student_data': student,
                    'student_text': self.student_texts[i],
                    'similarity': 1.0
                }]

        return []

    def get_context_for_query(self, query):
        """Return (context, used_flag). Only include context if we have a strict ID/name hit
           or a semantic hit with similarity >= SIM_THRESHOLD."""
        # 1) Try strict ID / exact full-name match
        strict_hits = self.search_by_id_or_name(query)
        if strict_hits:
            ctx_parts = []
            for student_info in strict_hits[:2]:
                ctx_parts.append(f"Student Information:\n{student_info['student_text']}\n")
            return "\n".join(ctx_parts).strip(), True

        # 2) Fall back to semantic search with threshold
        sem_hits = self.find_relevant_students(query, top_k=2)
        if sem_hits:
            ctx_parts = []
            for student_info in sem_hits:
                ctx_parts.append(f"Student Information:\n{student_info['student_text']}\n")
            return "\n".join(ctx_parts).strip(), True

        # 3) No confident hits → no context
        return "", False


# Initialize RAG system
student_rag = StudentRAG()


def initialize_rag_system():
    """Initialize the RAG system with student data"""
    print("Loading student data and creating embeddings...")
    student_rag.load_student_data()
    print(f"RAG system initialized with {len(student_rag.students_data)} students")


def chat_with_deepseek2(prompt, context=""):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Combine context with the user prompt if context is provided
    enhanced_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": enhanced_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 700
    }

    try:
        response = requests.post(api_url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"[EM]neutral[/EM][TEXT]حدث خطأ: {str(e)}[/TEXT]"


def chat_with_rag(user_question):
    """Enhanced chat function that uses RAG for student queries"""
    # Get relevant context
    context = student_rag.get_context_for_query(user_question)

    # Generate response using DeepSeek with context
    response = chat_with_deepseek2(user_question, context)
    return response


def check_for_chatbot_questions():
    data = readData("chatbot")
    print("Checking for update")
    for field in data:
        if field.get('Q') is not None:
            print("Question was found")
            Q = field['Q']
            isStudent = field['isStudent']
            if isStudent:
                students = readData("users")
                student = students[26]

                prom = f"""You are StudyMate for the student with the following information {student}, IMPORTANT NOTE: DO NOT MENTION ANY OF THE STUDENT INFORMATION, just utilize it to answer his question.

                 Answer question (Q) in one reply; no follow-ups.
If Q is vague or has multiple asks, pick the main one, state one brief assumption, then answer.
Match Q’s language. Be concise and supportive.
Math/science: show key formula/steps with units. Coding: give minimal correct snippet with 1–3 comments. Writing: give a tiny outline/checklist.
Refuse unsafe requests and suggest a safer educational alternative.

Format:
1) Direct Answer (1–2 sentences)
2) Key Steps (• bullets)
3) Worked Example (if useful)
4) Pitfall
5) Next Step (optional)

Now answer Q:

Q = {Q}
"""
                print("he is a student")
                A = chat_with_deepseek(prom)
            else:
                A = generate_question_prompt(Q)
            try:
                doc_ref = db.collection("chatbot").document(str(field['id']))
                doc_ref.update({
                    'A': A,
                    "Previous_Question": Q,
                    "Q": firestore.DELETE_FIELD
                })
                print(f"Question {field['id']} has been answered.")
            except Exception as e:
                print(f"Error answering {field['id']}: {str(e)}")
        else:
            print(f"Question {field.get('id')} already has answered. Skipping.")
    return None


def generate_question_prompt(Q):
    """Generate response using RAG system"""

    Q = """

    أنت مستشار ذكي للمعلمين باسم بصيرة تشات بوت، تتكلم بلغة عفوية، نصوحة، وإيجابية.
    يمكنك الوصول إلى البيانات التالية: الاسم، العمر، الصف، المعدل الأكاديمي (GPA)، سمات الشخصية الخمس (Big Five)، أساليب التعلم، نقاط القوة والضعف، المهارات.
    استخدم فقط البيانات المتوفرة لتوليد إجابات دقيقة ومفيدة.
    عند السؤال عن طالب محدد، اعتمد على بياناته المقدمة.
    عند السؤال عن شعبة أو مجموعة، اعطِ إجابة عامة ومقنعة جداً.
    لا تطلب معلومات مفقودة؛ إذا لم تكن متوفرة، تجاوزها ولا تسأل المستخدم عنها.
    لا تطرح أي أسئلة على المستخدم.
    إذا لم تعرف الجواب بدقة، قدم إجابة عامة ومقنعة ومفيدة.
    لا تستخدم رموز خاصة للتنسيق نهائياً.
    اجعل كل رد نهائيًا ومكتملًا ولا يفتح الباب لأسئلة متابعة.
    عند ورود أي سؤال مشابه للأسئلة التالية، استخدم الإجابة الأقرب لها كما هي:
    من أنت؟ أنا بصيرة تشات بوت، مستشار ذكي للمعلمين يقدّم توصيات واستراتيجيات تعليمية مخصصة بناءً على بيانات الطلاب المتوفرة مثل الاسم، العمر، الصف، المعدل، سمات الشخصية، أساليب التعلم، ونقاط القوة والضعف.
    كيف أستفيد منك في تعليم طلابي؟ أقدّم خطط تعليمية، أنشطة تكييفية، أساليب تقييـم بديلة، واقتراحات للتفاعل الصفّي مـرتكزة على ملف كل طالب، أطمح لتمكينك مبصراً لكل طالب على حده.
    ما المشكلة التي يحلها المشروع؟ يسهل التكييف الفوري للتدريس حسب اختلافات الطلاب، يقلّل الوقت الضائع في التجريب، ويرفع فعالية التدريس عبر توصيات عملية قابلة للتنفيذ لكل طالب ولكل شعبة.
    ما الجديد أو المبتكر في المشروع؟ الابتكار في الدمج بين سمات الشخصية (Big Five) وأساليب التعلم لكل طالب في الصف، لاقتراح تدخلات تعليمية دقيقة ومباشرة بدلاً من حلول معيارية عامة، والتي ترفع كفاءة التعليم بشكل عالي.
    كيف تحمون خصوصية بيانات الطلاب؟ يعتمد النظام مبدأ أقل وصول: يستخدم بيانات مشفّرة، وصول محدود للمعلّمين المصرّح لهم، وتطبيق سياسات بياني محدود متوافقة مع سياسات المؤسسة التعليمية.
    كيف تقيسون التأثير والنتائج؟ نقيس مؤشرات قبلية وبعدية مثل تغيّر المعدل، تغير الشخصية ، تغير اساليب التعلم، رضا المعلم/الطالب، ونقدّم تحاليل مُجمّعة مخصصة لكل طالب ولكل شعبة قابلة للعرض كدليل على الفعالية.
    ما منهجية التوصيات التعليمية؟ نستخدم خوارزميات تعتمد على قواعد تربوية مثبتة: مزج أساليب تدريس تتناسب مع نمط التعلم وسمات الشخصية، مع اقتراح أنشطة عملية وتقويمات قصيرة قابلة للتطبيق فوراً.
    هل يمكن التخصيص لذوي الاحتياجات الخاصة؟ نعم، عند توفر بيانات عن الاحتياجات نقدم تعديلات منطقية ومحددة للأنشطة والتقييمات مع إشارة لحدود التدخّل التي تحتاج موارد إنسانية إضافية.
    ما المتطلبات التقنية والتشغيلية؟ قاعدة بيانات تعليمية، واجهة إدخال بيانات للمعلمين، نظام صلاحيات محدود، وبنية استضافة آمنة.
    لا حاجة لمعدات متخصصة للعمل اليومي.
    ما حدود البوت وما لا يقدمه؟ لا يقدم تشخيصاً طبياً أو نفسياً نهائيّاً، ولا يتخذ قرارات إدارية نهائية. يُقدّم توصيات تربوية ويُترك القرار النهائي للمعلّم أو الموجّه.
    كيف تتعاملون مع التحيّز في البيانات؟ نطبّق قواعد لتقليل الاعتماد المفرط على أي سمة مفردة ونعطي أوزاناً متوازنة. نوصي بمتابعة بشرية دورية وتدقيق عينات لتصحيح الانحياز عند الحاجة.
    ما الجدوى الاقتصادية أو العائد المتوقع؟ تخفيض وقت إعداد المواد وتخصيص التدريس يؤدي إلى تحسين النتائج وتقليل الجهد الإداري؛ العائد يقاس بتوفير وقت المعلمين وارتفاع نِسَب النجاح ومؤشرات الأداء المدرسي.
    كيف تضمنون سهولة الاستخدام للمعلّم؟ إجابات مختصرة قابلة للتطبيق مباشرًا، قوالب جاهزة للأنشطة، وإرشادات قصيرة خطوة بخطوة لا تتطلب تدريب تقني مع واجهة بسيطة لإدخال البيانات.
    ما الذي ستقوله للّجنة في مقطع تقديمي مدته 30 ثانية؟ أنا بصيرة تشات بوت، أقدّم توصيات تعليمية مخصصة فورياً اعتماداً على بيانات الطالب والشعبة، أختصر وقت التخطيط وأزيد من فعالية التعليم عبر استراتيجيات مبنية على سمات شخصية وأساليب تعلم موثوقة.
    """ + Q

    return chat_with_rag(Q)


# ====================================================================================================(صناعة الاسئلة للاسايمنت والاختبارات)=====================
# ====================Main Function=================================

def get_pending_exams(firebase_path) -> List[Dict]:
    """
    Retrieve AIExam collection documents that are missing the 'answer' field

    Returns:
        List[Dict]: List of exam documents that need processing
    """
    try:
        exams_ref = db.collection(firebase_path)
        # Query documents where answer field is missing or null

        docs = exams_ref.where("Answer", '==', None).stream()

        pending_exams = []
        for doc in docs:
            exam_data = doc.to_dict()
            exam_data['doc_id'] = doc.id
            pending_exams.append(exam_data)

        # Also check for documents where answer field doesn't exist
        all_docs = exams_ref.stream()
        for doc in all_docs:
            exam_data = doc.to_dict()
            if "Answer" not in exam_data:
                exam_data['doc_id'] = doc.id
                # Avoid duplicates
                if not any(existing['doc_id'] == doc.id for existing in pending_exams):
                    pending_exams.append(exam_data)

        print(f"📋 Found {len(pending_exams)} pending exams to process")
        return pending_exams

    except Exception as e:
        print(f"❌ Error retrieving pending exams: {str(e)}")
        return []


def download_document_from_storage(file_path: str) -> Optional[bytes]:
    """
    Download a document from Firebase Storage

    Args:
        file_path (str): Path to the file in Firebase Storage

    Returns:
        Optional[bytes]: File content as bytes, None if error
    """
    try:
        blob = bucket.blob(file_path)
        if not blob.exists():
            print(f"❌ File not found in storage: {file_path}")
            return None

        content = blob.download_as_bytes()
        print(f"✅ Successfully downloaded file: {file_path}")
        return content

    except Exception as e:
        print(f"❌ Error downloading file {file_path}: {str(e)}")
        return None


def upload_pdf_to_gemini(pdf_bytes: bytes, display_name: str = "student_book") -> Optional[object]:
    """
    Upload PDF to Gemini using File API for documents larger than 20MB
    or direct processing for smaller files

    Args:
        pdf_bytes (bytes): PDF content as bytes
        display_name (str): Display name for the uploaded file

    Returns:
        Optional[object]: Uploaded file object or processed content
    """
    try:
        file_size_mb = len(pdf_bytes) / (1024 * 1024)
        print(f"📄 Processing PDF ({file_size_mb:.2f} MB)")

        if file_size_mb > 20:
            # Use File API for large files (>20MB)
            print("📤 Uploading large PDF using File API...")
            pdf_io = io.BytesIO(pdf_bytes)

            uploaded_file = client.files.upload(
                file=pdf_io,
                config=dict(
                    mime_type='application/pdf',
                    display_name=display_name
                )
            )

            # Wait for file to be processed
            print("⏳ Waiting for file processing...")
            while True:
                file_info = client.files.get(name=uploaded_file.name)
                if file_info.state == 'ACTIVE':
                    print("✅ File processed successfully")
                    return uploaded_file
                elif file_info.state == 'FAILED':
                    print("❌ File processing failed")
                    return None
                else:
                    print(f"⏳ File state: {file_info.state}, waiting...")
                    time.sleep(5)
        else:
            # Process small files directly
            print("📄 Processing PDF directly (inline)")
            return types.Part.from_bytes(
                data=pdf_bytes,
                mime_type='application/pdf'
            )

    except Exception as e:
        print(f"❌ Error uploading PDF to Gemini: {str(e)}")
        return None


def get_audience_gpa(audience_type: str, target_id: str) -> float:
    """
    Retrieve GPA information based on audience type and target

    Args:
        audience_type (str): Type of audience (student, class, group)
        target_id (str): Specific identifier for the target

    Returns:
        float: Average GPA for the target audience
    """
    try:
        if audience_type.lower() == 'student':
            # FIX: execute the query and use to_dict()
            students = db.collection('users').stream()
            for student in students:
                data = student.to_dict() or {}
                if data.get('name') == target_id:
                    return float(data.get('GPAout5', 3.0))

        elif audience_type.lower() == 'class':
            # Get class average GPA
            students_ref = db.collection('users').where('clas_name', '==', target_id)
            students = students_ref.stream()

            gpas = []
            for student in students:
                student_data = student.to_dict()
                if 'GPAout5' in student_data:
                    gpas.append(float(student_data['GPAout5']))
                    print(gpas)
            return sum(gpas) / len(gpas) if gpas else 3.0

        elif audience_type.lower() == 'group':
            # Get group average GPA
            students_ref = db.collection('users').where('group_name', '==', target_id)
            students = students_ref.stream()

            gpas = []
            for student in students:
                student_data = student.to_dict()
                if 'gpa' in student_data:
                    gpas.append(float(student_data['gpa']))

            return sum(gpas) / len(gpas) if gpas else 3.0

        return 3.0  # Default GPA

    except Exception as e:
        print(f"❌ Error retrieving GPA for {audience_type} {target_id}: {str(e)}")
        return 3.0


def generate_questions_with_gemini(pdf_content: object, num_templates: int,
                                   instruction: str, avg_gpa: float, firebase_path: str) -> Optional[str]:
    """
    Generate exam questions using Google Gemini API with PDF content

    Args:
        pdf_content: PDF content (either Part object for small files or uploaded file for large files)
        num_templates (int): Number of question templates to generate
        instruction (str): Additional instructions from teacher
        avg_gpa (float): Average GPA of target audience

    Returns:
        Optional[str]: Generated questions as formatted text
    """
    try:
        # Determine difficulty level based on GPA
        if avg_gpa >= 3.5:
            difficulty = "advanced"
        elif avg_gpa >= 2.5:
            difficulty = "intermediate"
        else:
            difficulty = "basic"

        if firebase_path == 'AIExam':
            promp = 'long exam'
        else:
            promp = 'short assignment' \
                # Construct the prompt
        prompt_text = f"""

You are an expert educational content creator. Your task is to generate test questions based on the provided student book content.

Create {num_templates} {promp} templates using ONLY the instructional parts of the attached student book, each template must use all units/chapters equally of the textbook unless otherwise specified
Target difficulty: {difficulty}.
Use The Teacher notes as follow when generating the questions: {instruction if instruction else "None"}

In addition to the Teacher notes, Follow STRICTLY the following rules when generating the questions:
- USE: lessons/units/topics, explanations, worked examples, figures, exercises, end-of-chapter questions.
- IGNORE: any front matter or metadata (preface/foreword, overview/purpose, edition/publisher/author pages, ISBN, policies).
- START: question generation after the first unit/chapter first page. DO NOT include any content before that
- DO NOT: ask for missing info; if something isn’t in the instructional content, skip it.
- DO NOT Use Special Characters for formatting.
- MAKE: the questions have variety of types, unless otherwise stated.
- ADD: A blank (new line) between questions, and 3 blanks (3 new lines) between templates
- Provide: Answers for each question

"""

        # Prepare content for API call
        if hasattr(pdf_content, 'name'):
            # Large file uploaded via File API
            print("🤖 Generating questions using uploaded file...")
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt_text, pdf_content]
            )
        else:
            # Small file processed directly
            print("🤖 Generating questions using inline PDF...")
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[prompt_text, pdf_content]
                )
            except Exception as e:
                print(e)
        if response and hasattr(response, 'text') and response.text:
            print(f"✅ Successfully generated {num_templates} question templates")
            return response.text
        else:
            print("❌ No response text received from Gemini API")
            return None

    except Exception as e:
        print(f"❌ Error generating questions with Gemini: {str(e)}")
        return None


def update_exam_with_answer(doc_id: str, answer: str, firebase_path: str) -> bool:
    """
    Update the AIExam document with the generated answer

    Args:
        doc_id (str): Document ID in AIExam collection
        answer (str): Generated questions/answers

    Returns:
        bool: True if successful, False otherwise
    """
    try:

        exam_ref = db.collection(firebase_path).document(doc_id)
        exam_ref.update({"Answer": answer})
        print(f"✅ Successfully updated exam document: {doc_id}")
        return True

    except Exception as e:
        print(f"❌ Error updating exam document {doc_id}: {str(e)}")
        return False


def process_single_exam(firebase_path, exam_data: Dict) -> bool:
    """
    Process a single exam request from start to finish

    Args:
        exam_data (Dict): Exam data from Firestore

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if firebase_path == "AIExam":
            instra = 'insrction'
            templates = 'NumOfTem'
        else:
            instra = 'instruction'
            templates = 'NumOfTem'
        doc_id = exam_data['doc_id']
        num_templates = min(exam_data.get(templates), 5)  # Max 5 as per requirements
        exams = readData(firebase_path)
        instruction = exam_data.get(instra)
        audience_type = exam_data.get('Type', 'student')
        target_id = exam_data.get('Target')
        print(instruction)
        print(f"\n🔄 Processing exam: {doc_id}")
        print(f"   Templates: {num_templates}, Audience: {audience_type}, Target: {target_id}")

        # Get document path
        document_path = "English_Textbook.pdf"

        if not document_path:
            print("❌ Could not determine document path")
            return False

        print(f"   Document path: {document_path}")

        # Download document from storage
        pdf_bytes = download_document_from_storage(document_path)
        if not pdf_bytes:
            return False

        # Upload/process PDF with Gemini
        pdf_content = upload_pdf_to_gemini(pdf_bytes, f"exam_{doc_id}")
        if not pdf_content:
            return False

        # Get audience GPA
        avg_gpa = get_audience_gpa(audience_type, target_id)
        print(f"   Target GPA: {avg_gpa:.2f}")

        # Generate questions
        generated_answer = generate_questions_with_gemini(
            pdf_content, num_templates, instruction, avg_gpa, firebase_path
        )

        if not generated_answer:
            return False

        # Update document with answer
        success = update_exam_with_answer(doc_id, generated_answer, firebase_path)

        # Clean up uploaded file if it was a large file
        if hasattr(pdf_content, 'name'):
            try:
                client.files.delete(name=pdf_content.name)
                print(f"🗑️ Cleaned up uploaded file: {pdf_content.name}")
            except Exception as e:
                print(f"⚠️ Could not clean up file: {str(e)}")

        return success

    except Exception as e:
        print(f"❌ Error processing exam {exam_data.get('doc_id', 'unknown')}: {str(e)}")
        return False


def process_all_pending_exams(firebase_path):
    """
    Main function to process all pending exam requests
    """
    try:
        print("🚀 Starting AI Exam Question Generator with Gemini Document Processing...")

        # Get all pending exams
        pending_exams = get_pending_exams(firebase_path)

        if not pending_exams:
            print("✅ No pending exams to process")
            return

        for exam in pending_exams:
            print(f"\n{'=' * 60}")
            if process_single_exam(firebase_path, exam):
                print(f"✅ Exam {exam['doc_id']} processed successfully")
                print("⏳ Waiting 3 seconds before next exam...")
                time.sleep(3)
            else:
                print(f"❌ Exam {exam['doc_id']} failed to process")

                # Add delay between requests to avoid rate limiting
                print("⏳ Waiting 3 seconds before next exam...")
                time.sleep(3)



    except Exception as e:
        print(f"❌ Error in main processing: {str(e)}")


def test_gemini_connection() -> bool:
    """
    Test the Gemini API connection

    Returns:
        bool: True if connection is successful
    """
    try:
        print("🧪 Testing Gemini API connection...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=["Hello, this is a test. Please respond with 'Connection successful'."]
        )

        if response and hasattr(response, 'text'):
            print(f"✅ Gemini API connection successful")
            print(f"   Response: {response.text[:100]}...")
            return True
        else:
            print("❌ Gemini API connection failed - no response")
            return False

    except Exception as e:
        print(f"❌ Gemini API connection failed: {str(e)}")
        return False


# ====================================================================================================(تصحيح للاسايمنت والاختبارات)=====================
# ====================Main Function=================================
def check_for_grading_tasks():
    """Check AIGrading collection for ungraded students and process them"""
    data = readData("AIGrading")
    print("Checking for grading tasks...")

    for document in data:
        print(f"Processing document ID: {document.get('id')}")

        # Check each student slot (1-5)
        for n in range(1, 6):
            student_name_field = f"st{n}_name"
            student_grade_field = f"st{n}_grade"
            student_inst_field = f"st{n}_inst"
            student_exam_field = f"st{n}_exam"

            # Check if student exam exists and grade is not available
            student_name = document.get(student_name_field)
            student_grade = document.get(student_grade_field)
            student_exam = document.get(student_exam_field)

            # Skip if exam is empty or doesn't exist
            if not student_exam or (isinstance(student_exam, list) and len(student_exam) == 0):
                continue

            if student_grade is None:

                print(f"Found ungraded student: {student_name} in slot {n}")

                # Get required data
                instruction = document.get(student_inst_field, "")
                exam_pictures = document.get(student_exam_field, [])
                answer_sheet = document.get("Answer_Sheet", "")
                student_name = document.get(student_name_field, f"Student {n}")  # Fallback name

                # Generate grade using AI
                grade = generate_grade_with_gemini(instruction, exam_pictures, answer_sheet)

                # Update the database
                try:
                    doc_ref = db.collection("AIGrading").document(str(document['id']))
                    doc_ref.update({
                        student_grade_field: grade
                    })
                    print(f"Student {student_name} (slot {n}) has been graded successfully.")
                except Exception as e:
                    print(f"Error grading student {student_name} (slot {n}): {str(e)}")
            else:
                if student_name and student_name.strip() != "":
                    print(f"Student {student_name} in slot {n} already graded. Skipping.")


def download_image_bytes(url):
    """Download image from URL and return bytes"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error downloading image from {url}: {str(e)}")
        return None


def analyze_exam_images_with_gemini(image_urls, image_type="exam"):
    """Analyze exam or answer sheet images using Gemini Vision API"""
    try:
        contents = []

        # Add all images to the content
        for url in image_urls:
            img_bytes = download_image_bytes(url)
            if img_bytes:
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

        # Add instruction based on image type
        if image_type == "exam":
            instruction = "Please analyze these exam images and extract all the questions and student answers clearly. Provide a detailed transcription of what you see in each image, including both the questions and the student's handwritten responses."
        else:  # answer_sheet
            instruction = "Please analyze this answer sheet image and extract all the correct answers, solutions, and marking criteria clearly. Provide a detailed transcription of what you see, including question numbers, correct answers, and any grading rubrics or point allocations."

        contents.append(instruction)

        # Generate content using Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )

        return response.text if response.text else f"Could not analyze the {image_type} images"

    except Exception as e:
        print(f"Error analyzing {image_type} images with Gemini: {str(e)}")
        return f"Error occurred while analyzing {image_type} images"


def generate_grade_with_gemini(instruction, exam_images, answer_sheet):
    """Generate grade using Gemini API with vision capabilities"""

    try:
        # First, analyze the exam images to extract content
        print("Analyzing exam images...")
        exam_content = analyze_exam_images_with_gemini(exam_images, "exam")

        # Check if answer_sheet is a URL (image) or text
        if isinstance(answer_sheet, str) and (answer_sheet.startswith('http') or 'firebasestorage' in answer_sheet):
            # Answer sheet is an image URL
            print("Analyzing answer sheet image...")
            answer_sheet_content = analyze_exam_images_with_gemini([answer_sheet], "answer_sheet")
        else:
            # Answer sheet is text or list of URLs
            if isinstance(answer_sheet, list):
                # Multiple answer sheet images
                print("Analyzing multiple answer sheet images...")
                answer_sheet_content = analyze_exam_images_with_gemini(answer_sheet, "answer_sheet")
            else:
                # Text-based answer sheet
                answer_sheet_content = str(answer_sheet)
        GRADING_SYSTEM_PROMPT = """
        You are a teacher with 30 years of experience in grading exams and assignments. Your task is to assess a student's submission using the provided ANSWER SHEET as the main reference. Your grading must be fair, criterion-referenced, and based on evidence.

        INSTRUCTIONS:
        1. Read the ANSWER SHEET carefully to understand correct answers, key steps, and marking scheme.
        2. Review the STUDENT SUBMISSION question by question.
        3. Allocate points fairly out of 100:
           - Give partial credit for correct reasoning or steps, even if the final answer is wrong.
           - Deduct points proportionally, not excessively.
        4. For each question at the feedback, write:
           - What the student did correctly.
           - State the mistakes or missing parts clearly.
           - One clear tip to improve.
        5. After grading all questions, provide:
           - Final Score: X/100
           - A short overall comment (2–3 sentences) summarizing performance.
           - Top 2–3 improvement steps for future exams.

        GRADING STYLE:
        - Be firm but encouraging, professional, and clear.
        - Avoid vague words like "good" or "bad". Be specific: e.g., "Missed unit conversion in Q3."
        - Use short, simple sentences.
        - Follow evidence-based feedback principles: clarify what was correct, what needs improvement, and how to improve.
        - Do not use special characters for formatting on the comments
        - Use third person language

        OUTPUT FORMAT:
        - Final Score: __/100
        - Overall Comment: (2–3 sentences)
        - Question Feedback:
          Q1: (comment)
          Q2: (comment)
          ...
        - Top Improvement Steps:
          1. ...
          2. ...
          3. ...

        Do not include extra explanations or metadata outside this format. 
        """  # Prepare the complete grading prompt
        grading_prompt = f"""
{GRADING_SYSTEM_PROMPT}

ANSWER SHEET (CORRECT SOLUTIONS):
{answer_sheet_content}

TEACHER'S GRADING INSTRUCTIONS:
{instruction}

STUDENT'S EXAM SUBMISSION (EXTRACTED FROM IMAGES):
{exam_content}

Please grade this student's exam based on the answer sheet and teacher's instructions provided above.
"""

        # Generate the grade using Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[grading_prompt],
        )

        return response.text if response.text else "Could not generate grade"

    except Exception as e:
        print(f"Error generating grade with Gemini: {str(e)}")
        return f"Error occurred while grading: {str(e)}"


# =======================(ارسال طلب تجميع قروبات لديب سيك واخذ اجابة منه)=====================
def coll_students(GroupType, classgroup, descr):
    # Filter students by class
    records = readData("users")
    # Required fields for each student
    jesus_record = records[26]
    students = []
    for record in records:
        if record['name'] == jesus_record['name']:
            jesus = {
                'clas_name': record['clas_name'],
                'name': record['name'],
                'gpa': record['GPAout5'],
                'Extraversion': record['Big1'] * 5,
                'Agreeableness': record['Big2'] * 5,
                'Conscientiousness': record['Big3'] * 5,
                'Neuroticism': record['Big4'] * 5,
                'Openness': record['Big5'] * 5,
                'inferencing': record['LSResult'][0],
                'note_taking': record['LSResult'][1],
                'dictionary_use': record['LSResult'][2],
                'oral_repetition': record['LSResult'][3],
                'activation': record['LSResult'][4],
                'visual_auditory_encoding': record['LSResult'][5],
                'structural_encoding': record['LSResult'][6],
                'self_inflation': record['LSResult'][7],
                'selective_attention': record['LSResult'][8]
            }
            continue
        student = {
            'clas_name': record['clas_name'],
            'name': record['name'],
            'gpa': record['GPAout5'],
            'Extraversion': record['Big1'] * 5,
            'Agreeableness': record['Big2'] * 5,
            'Conscientiousness': record['Big3'] * 5,
            'Neuroticism': record['Big4'] * 5,
            'Openness': record['Big5'] * 5,
            'inferencing': record['LSResult'][0],
            'note_taking': record['LSResult'][1],
            'dictionary_use': record['LSResult'][2],
            'oral_repetition': record['LSResult'][3],
            'activation': record['LSResult'][4],
            'visual_auditory_encoding': record['LSResult'][5],
            'structural_encoding': record['LSResult'][6],
            'self_inflation': record['LSResult'][7],
            'selective_attention': record['LSResult'][8]
        }
        students.append(student)
    class_students = [student for student in students if student['clas_name'] == classgroup]

    if not class_students:
        return {"Notes": "No students found in this class", "CM": {}}

    print(jesus['name'])
    # Create strict prompt for DeepSeek API

    prompt = f"""
You are an intelligent matcher designed to recommend the single best group-mate for a given student.  
Your decisions are based on: Big Five personality traits, learning styles, GPA, the desired group type, and the student's additional description.  
You must remain objective, deterministic, and concise in your reasoning.

GOAL:
Identify ONE best match from the "class_students" list for the given "student_profile".  

STRICT RULES:
1. DO NOT mention or reveal any GPA, traits, or learning styles of any students.  
2. Provide only the name of the best match and a short justification based on compatibility with:
   - The student's description
   - The desired group type
3. Be deterministic and concise (no vague or probabilistic language).  
4. Do not include numbers in either the header nor the body
5. DO NOT: ask for missing info; if something isn’t in the instructional content, skip it.
6. DO NOT Use Special Characters for formatting.

OUTPUT FORMAT (strict):
Best Match: [Full Name]

Justification: [1–2 sentences explaining why this match is ideal without exposing private details.]

INPUTS:
- student_profile: {jesus}
- group_type: {GroupType}
- student_description: {descr}
- class_students: {class_students}
    """

    print(prompt)
    # Call the API (simulated in this example)
    try:
        result = chat_with_deepseek(prompt)
        # The first part is everything before '<'
        groupMembers = result

        return groupMembers
    except Exception as e:
        print(e)
        groupMembers, Notes = "", ""
        return groupMembers, Notes


def coll_update_check():
    groups = readData("CM")
    for group in groups:
        # Check if 'GroupMembers' key is missing (not just empty/None)
        if group.get('Answer') is None:  # or 'GroupMembers' not in group
            GroupType = group['GroupType']
            classgroup = "Class 1"
            descr = group['descr']

            # Generate new members and notes
            new_member = coll_students(GroupType, classgroup, descr)

            try:
                # Update only the missing fields in Firestore
                doc_ref = db.collection("CM").document(str(group['id']))
                doc_ref.update({
                    'Answer': new_member
                })
                print(f"Updated CM {group['id']} with new mate.")
            except Exception as e:
                print(f"Error updating group {group['id']}: {str(e)}")
        else:
            print(f"colleagues at {group.get('id')} already has members. Skipping.")


# =========================================(تحديث معلومات الطلاب بناء على ملاحظة الاستاذ)============================================

def analyze_comment_and_update_parameters(teacher_comment, current_parameters):
    """
    Analyze teacher comment about student behavior and predict personality trait changes
    Returns a dictionary with updated Big5 and Learning Styles parameters
    """

    prompt = f"""
    As a psychological assessment AI, analyze the following teacher's observation about a student's behavior and predict how this can influence their personality traits and learning styles if any.
    DO NOT UPDATE ANY PARAMETER UNLESS IT IS ULTIMETALY AFFECTED BY THE TEACHER'S COMMENT.

    Teacher's Behavioral Observation: {teacher_comment}

    Current Student Parameters:
    Big Five Personality Traits (0.0-1.0 scale):
    - Big1 (Openness): {current_parameters['big5']['Big1']}
    - Big2 (Conscientiousness): {current_parameters['big5']['Big2']}
    - Big3 (Extraversion): {current_parameters['big5']['Big3']}
    - Big4 (Agreeableness): {current_parameters['big5']['Big4']}
    - Big5 (Neuroticism): {current_parameters['big5']['Big5']}

    Current Learning Styles:
    - inferencing: {current_parameters['learning_styles']['results'][0] if len(current_parameters['learning_styles']['results']) > 0 else 0}
    - note_taking: {current_parameters['learning_styles']['results'][1] if len(current_parameters['learning_styles']['results']) > 1 else 0}
    - dictionary_use: {current_parameters['learning_styles']['results'][2] if len(current_parameters['learning_styles']['results']) > 2 else 0}
    - oral_repetition: {current_parameters['learning_styles']['results'][3] if len(current_parameters['learning_styles']['results']) > 3 else 0}
    - activation: {current_parameters['learning_styles']['results'][4] if len(current_parameters['learning_styles']['results']) > 4 else 0}
    - visual_auditory_encoding: {current_parameters['learning_styles']['results'][5] if len(current_parameters['learning_styles']['results']) > 5 else 0}
    - structural_encoding: {current_parameters['learning_styles']['results'][6] if len(current_parameters['learning_styles']['results']) > 6 else 0}
    - self_inflation: {current_parameters['learning_styles']['results'][7] if len(current_parameters['learning_styles']['results']) > 7 else 0}
    - selective_attention: {current_parameters['learning_styles']['results'][8] if len(current_parameters['learning_styles']['results']) > 8 else 0}

    Based on this behavioral observation and the student's current psychological profile, provide incremental updates to reflect how this behavior might influence their personality and learning approach.

    Return ONLY the parameters that need updating as a JSON object with:
    - "big5_updates": object with keys like "Big1", "Big2", etc. (only include traits that should change)
    - "learning_styles_updates": object with index numbers as strings "0", "1", etc. (only include styles that should change)

    Make realistic, small incremental changes (±0.05 to ±0.3 for personality, ±0.5 to ±2.0 for learning styles) based on the observed behavior.
    """

    # Call your AI service here (ChatGPT, DeepSeek, etc.)
    response = chat_with_deepseek(prompt)

    try:
        # Parse the AI response to get parameter updates
        updates = json.loads(response)
        return updates
    except json.JSONDecodeError:
        print(f"⚠️ فشل في تحليل رد الذكي الاصطناعي: {response}")
        return None


def update_student_parameters(user_id, parameter_updates, current_parameters, update_percentage=0.2):
    """
    Update student's Big5 and Learning Styles parameters in Firebase
    """
    try:
        doc_ref = db.collection('users').document(user_id)

        # Prepare the update data
        update_data = {}

        # Update Big5 parameters with percentage control
        if 'big5_updates' in parameter_updates and parameter_updates['big5_updates']:
            for param, suggested_value in parameter_updates['big5_updates'].items():
                if isinstance(suggested_value, (int, float)) and 0.0 <= suggested_value <= 1.0:
                    old_value = current_parameters['big5'].get(param, 0.5)
                    new_value = old_value + (suggested_value - old_value) * update_percentage
                    new_value = max(0.0, min(1.0, new_value))  # Ensure within bounds

                    update_data[param] = new_value
                    print(f"📊 تحديث {param}: {old_value:.3f} → {new_value:.3f} (اقتراح: {suggested_value:.3f})")

        # Update Learning Styles parameters with percentage control
        if 'learning_styles_updates' in parameter_updates and parameter_updates['learning_styles_updates']:
            current_ls_result = current_parameters['learning_styles']['results'].copy()

            for index_str, suggested_value in parameter_updates['learning_styles_updates'].items():
                try:
                    index = int(index_str)
                    if 0 <= index < len(current_ls_result) and isinstance(suggested_value, (int, float)):
                        old_value = current_ls_result[index]
                        new_value = old_value + (suggested_value - old_value) * update_percentage

                        current_ls_result[index] = new_value
                        print(
                            f"📚 تحديث أسلوب التعلم [{index}]: {old_value:.2f} → {new_value:.2f} (اقتراح: {suggested_value:.2f})")
                except (ValueError, IndexError):
                    print(f"⚠️ فهرس أسلوب التعلم غير صحيح: {index_str}")

                update_data['LSResult'] = current_ls_result
        print(update_data)
        # Perform the update
        if update_data:
            doc_ref.update(update_data)
            print(f"✅ تم تحديث معاملات المستخدم {user_id}")
            print(f"🔄 البيانات المحدثة: {update_data}")
        else:
            print(f"ℹ️ لا توجد معاملات للتحديث للمستخدم {user_id}")

    except Exception as e:
        print(f"❌ فشل في تحديث معاملات المستخدم {user_id}: {e}")


# Optional: Function to get current student parameters (required for analysis)
def get_current_student_parameters(user_id):
    """
    Retrieve current student parameters for behavioral analysis
    """
    try:
        doc_ref = db.collection('users').document(user_id)
        doc = doc_ref.get()

        if doc.exists:
            data = doc.to_dict()

            # Extract Big5 parameters
            big5_params = {
                'Big1': data.get('Big1', 0.5),  # Default to neutral values if not set
                'Big2': data.get('Big2', 0.5),
                'Big3': data.get('Big3', 0.5),
                'Big4': data.get('Big4', 0.5),
                'Big5': data.get('Big5', 0.5)
            }

            # Extract Learning Styles parameters
            ls_result = data.get('LSResult', [2.5] * 9)  # Default neutral values
            ls_names = data.get('LSnames', [
                "inferencing", "note_taking", "dictionary_use",
                "oral_repetition", "activation", "visual_auditory_encoding",
                "structural_encoding", "self_inflation", "selective_attention"
            ])

            return {
                'big5': big5_params,
                'learning_styles': {
                    'results': ls_result,
                    'names': ls_names
                }
            }
        else:
            print(f"❌ المستخدم {user_id} غير موجود")
            return None

    except Exception as e:
        print(f"❌ فشل في استرداد معاملات المستخدم {user_id}: {e}")
        return None


def on_user_data_changed(user_id, ask):
    """
    Updated function to analyze teacher behavioral observations and update student parameters
    """
    # البحث عن مفتاح 'Ask' الذي يحتوي على تعليق المعلم
    teacher_comment = ask

    if teacher_comment:
        print(f"\n🎯 تحليل ملاحظة المعلم حول سلوك الطالب: {teacher_comment}")

        # Get current student parameters first
        current_parameters = get_current_student_parameters(user_id)

        if current_parameters:
            # Analyze the teacher's behavioral observation and get parameter updates
            parameter_updates = analyze_comment_and_update_parameters(teacher_comment, current_parameters)

            if parameter_updates:
                # Update student parameters based on behavioral analysis
                update_student_parameters(user_id, parameter_updates, current_parameters)
            else:
                print(f"⚠️ فشل في تحليل ملاحظة المعلم للمستخدم {user_id}")
        else:
            print(f"⚠️ فشل في استرداد المعاملات الحالية للمستخدم {user_id}")

        # حذف تعليق المعلم من فير بيس بعد التحليل والتحديث
        try:
            doc_ref = db.collection('users').document(user_id)
            doc_ref.update({
                'Ask': firestore.DELETE_FIELD  # حذف الحقل بعد معالجة ملاحظة المعلم
            })
            print(f"🧹 تم حذف ملاحظة المعلم من المستخدم {user_id}")
        except Exception as e:
            print(f"⚠️ فشل حذف ملاحظة المعلم: {e}")
    else:
        print(f"ℹ️ لا توجد ملاحظة جديدة من المعلم للمستخدم {user_id}")


def ask_update():
    data = readData("users")
    print("Checking for comments")
    for field in data:
        if field.get('Ask') is not None:
            print("Question was found")
            on_user_data_changed(field['id'], field['Ask'])
            try:
                doc_ref = db.collection("users").document(str(field['id']))
                doc_ref.update({
                    "Ask": firestore.DELETE_FIELD
                })
                print(f"comment {field['id']} has been resolved.")
            except Exception as e:
                print(f"Error answering {field['id']}: {str(e)}")


# =======================( قراءة البيانات)========================================================================================================================
def readData(dataname):
    result = []
    users_ref = db.collection(dataname)
    docs = users_ref.stream()

    for doc in docs:
        user_data = doc.to_dict()
        user_data['id'] = doc.id
        result.append(user_data)
    return result


# =======================(ارسال طلب لديب سيك واخذ اجابة منه)=====================
def chat_with_deepseek(prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 700
    }
    try:
        response = requests.post(api_url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"[EM]neutral[/EM][TEXT]حدث خطأ: {str(e)}[/TEXT]"


# =======================(الاستماع للتحديثات من فير بيس كل 5 ثواني)=====================
previous_users_data = {}


def check_for_changes():
    global first_time
    group_update_check()
    check_for_chatbot_questions()
    process_all_pending_exams('AIExam')
    process_all_pending_exams('AIAsimnt')
    check_for_grading_tasks()
    coll_update_check()
    ask_update()
    now = datetime.now()
    # Check if it's ~midnight in UTC+3 (which is 21:00 UTC)
    if now.hour == 21 and first_time == True:
        generate_combined_averages_prompt()
        first_time = False
    elif now.hour == 22:
        first_time = True

    global previous_users_data


# =======================(بداية البرنامج)=====================
if __name__ == "__main__":
    print("📡 جاري مراقبة تغييرات Firestore...\n")
    first_time = True
    initialize_rag_system()  # <-- add this
    try:
        while True:
            check_for_changes()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف البرنامج يدويًا.")
