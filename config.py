"""
Configuration file for data processing and model training
"""

# Mapping alias for standardizing column names
COLUMN_MAPPING = {
    # Student ID variations
    'MSSV': 'student_id',
    'Student_ID': 'student_id',
    
    # Name variations
    'Họ tên': 'full_name',
    'Họ và tên': 'full_name',
    'name': 'full_name',
    'LastName': 'last_name',
    'FirstName': 'first_name',
    'FullName': 'full_name',
    
    # Subject variations
    'Mã môn học': 'subject_id',
    'Subject_ID': 'subject_id',
    'Tên môn học': 'subject_name',
    'Subject_Name': 'subject_name',
    
    # Score variations
    'Điểm': 'score',
    'exam_score': 'exam_score',
    'summary_score': 'summary_score',
    'summary_score': 'final_score',
    
    # Attendance
    'Điểm danh': 'attendance_status',
    'Ngày đi học': 'attendance_date',
    'Buổi': 'session',
    
    # Academic info
    'Niên khoá': 'academic_year',
    'Học kì': 'semester',
    'year': 'academic_year',
    'semester': 'semester',
    'semester_year': 'academic_year',
    
    # Class info
    'Class_ID': 'class_id',
    'Class_Id': 'class_id',
    
    # Conduct score
    'conduct_score': 'conduct_score',
    'student_conduct_classification': 'conduct_classification',
    
    # Study hours
    'accumulated_study_hours': 'study_hours',
    'accumulated_study_minutes': 'study_minutes',
    'time': 'study_time',
    
    # Pass/Fail
    'Passed_the_module': 'passed_module',
    
    # Other common fields
    'Lecturer_Name': 'lecturer_name',
    'Major_Name': 'major_name',
    'Faculty_Name': 'faculty_name',
    'Credit_Hours': 'credit_hours',
}

# Key columns for enrollment table
ENROLLMENT_KEY_COLUMNS = ['student_id', 'subject_id', 'academic_year', 'semester']

# Target variable
TARGET_COLUMN = 'pass_fail'

# Output directory structure
OUTPUT_DIRS = {
    'models': 'output/models',
    'predictions': 'output/predictions',
    'metrics': 'output/metrics',
    'charts': 'output/charts',
    'data': 'output/data'
}



