from flask import Flask, render_template, request, redirect
import pandas as pd
from transformers import pipeline

app = Flask(__name__)

def analyze_sentiment(text):
    sentiment_pipeline = pipeline("text-classification", model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")
    try:
        result = sentiment_pipeline(text)
        label = result[0]['label']
        if label.upper() == "POSITIVE":
            return "إيجابي"
        elif label.upper() == "NEGATIVE":
            return "سلبي"
        else:
            return "محايد"
    except Exception as e:
        return "محايد"

def extract_written_feedback(df):
    """
    استخراج التعليقات المكتوبة من الأعمدة الأخيرة وتحليل مشاعر كل تعليق باستخدام analyze_sentiment.
    تُعيد الدالة قاموسًا يحتوي على قائمة من القواميس، كل منها يحتوي على نص التعليق وتصنيفه.
    """
    feedback = {}
    writing_columns = df.columns[-3:]
    for col in writing_columns:
        raw_feedback = df[col].dropna().tolist()
        feedback[col] = []
        for item in raw_feedback:
            sentiment = analyze_sentiment(str(item))
            feedback[col].append({"text": item, "sentiment": sentiment})
    return feedback

def compute_weighted_satisfaction(df):
    """
    تُحسب الدالة المتوسط المرجّح لكل سؤال اعتمادًا على أوزان محددة:
      - "أوافق بشدة": 4
      - "أوافق": 3
      - "لا أوافق": 2
      - "لا أوافق بشدة": 1
    تُرجع الدالة قاموسًا يحتوي على المتوسط لكل عمود.
    """
    weights = {"أوافق بشدة": 4, "أوافق": 3, "لا أوافق": 2, "لا أوافق بشدة": 1}
    scores = {}
    for col in df.columns[6:-3]:
        total_weight = 0
        total_count = 0
        counts = df[col].value_counts().to_dict()
        for rating, weight in weights.items():
            count = counts.get(rating, 0)
            total_weight += count * weight
            total_count += count
        avg_score = total_weight / total_count if total_count > 0 else 0
        scores[col] = round(avg_score, 2)
    return scores

def analyze_data(df):
    """
    تقسيم تحليل الأسئلة إلى ثلاث مجموعات:
      - بداية المقرر
      - ما حدث خلال المقرر
      - التقويم العام
    يتم احتساب عدد إجابات "أوافق بشدة" لكل سؤال في كل مجموعة.
    """
    analysis = {}
    analysis['beginning_clarity'] = df.iloc[:, 6:9].apply(lambda x: (x == 'أوافق بشدة').sum(), axis=0).to_dict()
    analysis['during_clarity'] = df.iloc[:, 9:23].apply(lambda x: (x == 'أوافق بشدة').sum(), axis=0).to_dict()
    analysis['overall_satisfaction'] = df.iloc[:, 23:-3].apply(lambda x: (x == 'أوافق بشدة').sum(), axis=0).to_dict()
    return analysis

def calculate_overall_satisfaction(df):
    """
    تحسب الدالة متوسط التقييم المرجّح لكل طالب عبر جميع أسئلة التقييم،
    ثم تُحوّل الدرجة إلى نسبة رضا على مقياس من 100%.
    تعتمد المعادلة على أن أدنى تقييم هو 1 وأعلى تقييم هو 4.
    """
    weights = {"أوافق بشدة": 4, "أوافق": 3, "لا أوافق": 2, "لا أوافق بشدة": 1}
    total_score = 0
    total_responses = 0
    for col in df.columns[6:-3]:
        counts = df[col].value_counts().to_dict()
        for rating, weight in weights.items():
            count = counts.get(rating, 0)
            total_score += count * weight
            total_responses += count
    overall_avg = total_score / total_responses if total_responses > 0 else 0
    # تحويل المتوسط المرجّح إلى نسبة رضا: (المتوسط - 1) / (4 - 1) * 100
    satisfaction_percentage = (overall_avg - 1) / 3 * 100
    return round(satisfaction_percentage, 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)
        
        analysis_results = analyze_data(df)
        overall_satisfaction = calculate_overall_satisfaction(df)

        course_info = {
            'course_name': df.iloc[1, 2], 
            'program_name': df.iloc[1, 3],  
            'semester': df.iloc[1, 4],
            'year': df.iloc[1, 5],  
        }

        written_feedback = extract_written_feedback(df)
        weighted_scores = compute_weighted_satisfaction(df)

        feedback_clean = {key.strip(): value for key, value in written_feedback.items()}

        return render_template('result.html', 
                            analysis=analysis_results, 
                            course_info=course_info,
                            feedback=feedback_clean,
                            satisfaction=overall_satisfaction, 
                            weighted_scores=weighted_scores)
    else:
        return 'الرجاء تحميل ملف Excel صحيح.'

if __name__ == '__main__':
    app.run(debug=True)
