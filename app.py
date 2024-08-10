# import module
import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from underthesea import word_tokenize 
import regex as re
from stopwordsiso import stopwords
#from text_processing_utils import text_preprocess
#from text_processing_utils import colorize_words


############ 2. SETTING UP THE PAGE LAYOUT AND TITLE ############
# Call set_page_config() as the first Streamlit command in your script
st.set_page_config(page_title="Hệ thống Xác thực Phản ánh, Kiến nghị", page_icon="📢", layout="centered")




with open('styles.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


# Introduction section
st.sidebar.title('Trường Đại Học Sài Gòn')
st.sidebar.header('GVHD: TS.Vũ Ngọc Thanh Sang')
st.sidebar.header('HV: Lê Đăng Quang')
st.sidebar.markdown("""
- **Mã số học viên:** CH11221006
- **Lớp học:** KMTUD221
- **Khóa học:** 22.1
""")



# Title
st.title("HỆ THỐNG XÁC THỰC PHẢN ÁNH, KIẾN NGHỊ")

# Xử lý phân loại với mô hình đã load từ file
MODEL_PATH = "models"
# Load mô hình  từ file
with open(os.path.join(MODEL_PATH, "svm.pkl"), 'rb') as model_file:
    nb_model = pickle.load(model_file)
# Định nghĩa label encoder và khởi tạo nếu cần thiết
label_encoder = LabelEncoder()
sample_labels = ['__label__PA', '__label__KPA'] #['__label__DGYK', '__label__DL', '__label__KDTT', '__label__Khac', '__label__PA', '__label__TC', '__label__TGTP']    # Example labels
label_encoder.fit(sample_labels)

# # Load mô hình  từ file
# with open(os.path.join(MODEL_PATH, "svm-KPA.pkl"), 'rb') as KPA_model_file:
#     KPA_model = pickle.load(KPA_model_file)
# # Định nghĩa label encoder và khởi tạo nếu cần thiết
# KPA_label_encoder = LabelEncoder()
# KPA_sample_labels = ['_label__KDTT','__label__GDPL','__label__TC','__label__DGYK','__label__TGTP','__label__Khac','__label__DL'] #['__label__DGYK', '__label__DL', '__label__KDTT', '__label__Khac', '__label__PA', '__label__TC', '__label__TGTP']    # Example labels
# KPA_label_encoder.fit(KPA_sample_labels)


min_length = 50


# Form nhập phản hồi
document = st.text_area('Hãy nhập nội dung:'.format(min_length), height=180 )
submitted = st.button('Thực hiện')

# Hàm dự đoán
def predict_label(text):
    # Dự đoán với mô hình Naive Bayes
    label = nb_model.predict([text])[0]
    return label

# Hàm dự đoán
def predict_label_KPA(text):
    # Dự đoán với mô hình Naive Bayes
    label = KPA_model.predict([text])[0]
    return label

 
uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
 
def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
dicchar = loaddicchar()

# Hàm chuyển Unicode dựng sẵn về Unicde tổ hợp (phổ biến hơn)
def convert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']

nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)

def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
            return ''.join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            # for index2 in nguyen_am_index:
            #     if index2 != index:
            #         x, y = nguyen_am_to_ids[chars[index]]
            #         chars[index2] = bang_nguyen_am[x][0]
            return ''.join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
        else:
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
        # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
        # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
    return ''.join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def chuan_hoa_dau_cau_tieng_viet(sentence):
    """
        Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
        :param sentence:
        :return:
        """
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
        # print(cw)
        if len(cw) == 3:
            cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)

def remove_html(txt):
    return re.sub(r'<[^>]*>', '', txt)

vietnamese_stopwords = stopwords(['vi'])
def remove_stopwords(sentence, stopwords):
    tokens = sentence.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stopwords]
    return ' '.join(filtered_tokens)

def text_preprocess(document):
    # Process each cell in the dataframe to remove stopwords
    document = remove_stopwords(document, vietnamese_stopwords)
    # xóa html code
    document = remove_html(document)
    # chuẩn hóa unicode
    document = convert_unicode(document)
    # chuẩn hóa cách gõ dấu tiếng Việt
    document = chuan_hoa_dau_cau_tieng_viet(document)
    # tách từ
    document = word_tokenize(document, format="text")
    # đưa về lower
    document = document.lower()
    # xóa các ký tự không cần thiết
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',document)
    # xóa khoảng trắng thừa
    document = re.sub(r'\s+', ' ', document).strip()

    #document = remove_whitespace(document)
    return document








processed_document = text_preprocess(document)
st.subheader('Kết quả:')
# Xử lý nếu người dùng đã nhấn nút Phân Loại
if submitted:
    if document.strip() == '':
        st.warning('Vui lòng nhập nội dung.')
    else:
        # Xử lý dự đoán và hiển thị kết quả
        # Dự đoán nhãn
        predicted_label = label_encoder.inverse_transform([predict_label(processed_document)])
        if predicted_label == '__label__PA':
            st.success("Phản ánh, Kiến nghị hợp lệ")
        else:
            st.error("Phản ánh, Kiến nghị không hợp lệ")
             # Dự đoán nhãn KPA
           # predicted_label_KPA = KPA_label_encoder.inverse_transform([predict_label_KPA(processed_document)])[0]
            #st.write(predicted_label_KPA)
        #st.write(predicted_label)
      # Phát hiện và tô màu từ ngữ thô tục trong văn bản
    #if processed_document:           
           # result = processed_document
           # st.markdown(f'**Từ ngữ được đánh dấu:**\n{result}', unsafe_allow_html=True)
        #st.write(processed_document)


            
st.markdown("---")

st.header("Danh sách")
uploaded_file = st.file_uploader("Chọn một file Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
     
        st.success("Tải file và đọc thành công.")
        processed_document_row = df["Nội dung phản ánh"].str.lower()      
        def map_label(text):
            # Dự đoán với mô hình Naive Bayes
            label = nb_model.predict([text_preprocess(text)])[0]
            if label == 0:
                return "Không hợp lệ"
            elif label == 1:
                return "Hợp lệ"
            else:
                return "Không xác định"
        # Thêm cột "Xác thực" với giá trị mặc định (có thể thay đổi)        
        df['Xác thực'] = processed_document_row.apply(map_label)
        df = df.drop('STT', axis=1)  # Xóa cột 'id'    
      
        st.dataframe(df)
        # Perform any operations on the DataFrame here
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
else:
    st.warning("Không tải được file.")
