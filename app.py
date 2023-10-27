import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

st.set_page_config(
    page_title='Analisis Sentimen Berbasis Aspek pada Ulasan Produk Elektronik Laptop',
    layout='wide',
)

def set_outer_color(label):
    if label == 'BOD':
        return '#FEF9C3' # kuning muda
    elif label == 'POW':
        return '#DBEAFE' # biru muda
    elif label == 'KEY':
        return '#FAE8FF' # ungu muda (fuschia)
    elif label == 'POS':
        return '#DCFCE7' # hijau muda
    elif label == 'NEG':
        return '#FEE2E2' # merah muda (rose)

def set_outer_span(label):
    outer_span_start = '<span style="background-color: '+ set_outer_color(label) +'; padding: 4px; border-radius: 5px; margin: 5px; display: inline;">'
    return outer_span_start

def set_inner_color(label):
    if label == 'BOD':
        return '#EAB308' # kuning tua
    elif label == 'POW':
        return '#2563EB' # biru tua
    elif label == 'KEY':
        return '#C026D3' # ungu tua (fuschia)
    elif label == 'POS':
        return '#16A34A' # hijau tua
    elif label == 'NEG':
        return '#DC2626' # merah tua (rose)

def set_inner_span(label):
    inner_span_start = '<span style="background-color: '+ set_inner_color(label) +'; padding: 1px 6px; margin-left: 3px; border-radius: 5px; font-weight: bold; color: #fff; display: inline;">'
    return inner_span_start

def highlight_token(text, label):
    span_end = '</span>'
    idx = 0
    highlighted_text = []
    text_len = len(text)

    curr_asp = ''

    list_token = [] # untuk menyimpan akumulasi token-token sebelumnya
    status = 1 # default status (status = 1): menyimpan. status = 0: mencetak dengan span. status = 2: mencetak tanpa span.

    for token, token_label in zip(text, label):
        if idx == 0:
            prev_asp = ''
        else:
            prev_asp = curr_asp
        curr_asp = token_label

        if curr_asp == 'O':
            if prev_asp != 'O' and prev_asp != '':
                status = 0
            else:
                status = 2
        else:
            if prev_asp == 'O':
                status = 2
            else:
                status = 1
        idx += 1

        # TAMPUNG
        if status == 1: 
            list_token.append(token)
        # CETAK SPAN PADA INDEX SEBELUMNYA
        elif status == 0:  
            # lakukan membuat span
            tokens = " ".join(list_token)
            new_span = f'{set_outer_span(prev_asp)} {tokens} {set_inner_span(prev_asp)} {prev_asp} {span_end} {span_end}'
            highlighted_text.append(new_span)

            # reset ulang variabel
            list_token = []
            list_token.append(token)
            status = 1
        # CETAK TANPA SPAN PADA INDEX SEBELUMNYA
        else:
            for i in list_token:
                new_span = f'{i}'
                highlighted_text.append(new_span)
            
            # reset ulang variabel
            list_token = []
            list_token.append(token)
            status = 1

        # CETAK KHUSUS INDEX TERAKHIR
        if idx-1 == text_len-1:
            if curr_asp == 'O':
                new_span = f'{token}'
                highlighted_text.append(new_span)
            else:
                tokens = " ".join(list_token)
                new_span = f'{set_outer_span(curr_asp)} {tokens} {set_inner_span(curr_asp)} {curr_asp} {span_end} {span_end}'
                highlighted_text.append(new_span)

    result = ' '.join(highlighted_text)
    div = f"<div class='custom-div'>{result}</div>"
    return div

@st.cache_resource
def load_my_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_resource
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    handle.close()
    return tokenizer

@st.cache_data
def aspect_predict(_model, input):
    return _model.predict(input)

@st.cache_data
def sentiment_predict(_model, input):
    return _model.predict(input)

def get_output_format(tokenizer, sentence, model, max_length, tags, type):
    sentence = preprocess_sentence(sentence)
    sentence_seq = tokenizer.texts_to_sequences([sentence])
    sentence_padded = pad_sequences(sentence_seq, maxlen=max_length, padding='post', truncating='post')
    predictions = []

    if type == 'a': # 'a' for aspect
        predictions = aspect_predict(model, sentence_padded)[0]
    else: # 's' for sentiment
        predictions = sentiment_predict(model, sentence_padded)[0]

    predicted_tags = []
    for prediction in predictions:
        result = np.argmax(prediction)
        result_tag = tags[result]
        predicted_tags.append(result_tag)

    result = []
    for word, tag in zip(sentence.split(), predicted_tags):
        result.append(f'{word} {tag}')
    return result

def preprocess_sentence(original_sentence):
    # mencari tanda baca yang tidak diikuti spasi
    pattern = r'(?<=[,.!?\(\)-/\'\"▪️%~:;@#$^&*_+=])|(?=[,.!?\(\)-/\'\"▪️%~:;@#$^&*_+=])'
    new_sentence = re.sub(pattern, ' ', original_sentence)
    return new_sentence

def clear_input():
    st.session_state['text'] = ""

def set_input(text):
    st.session_state['text'] = text

def import_css(file_path):
    with open(file_path, "r") as f:
        css = f.read()
    f.close()
    return css 

def main():
    # ======================================================================== SET VARIABLE ========================================================================
    # load model
    aspect_model = load_my_model('resource/ASPECT-ONLY_EXPERIMENT-9')
    # sentiment_model = load_my_model('resource/SENTIMENT-ONLY_EXPERIMENT-4')
    sentiment_model = load_my_model('resource/SENTIMENT-ONLY_EXPERIMENT-6-c3')

    # load tokenizer
    tokenizer = load_tokenizer('resource/tokenizer.pickle')

    MAX_LENGTH = 600

    # tag
    aspect_tags = ['O', 'B-BOD', 'I-BOD', 'B-POW', 'I-POW', 'B-KEY', 'I-KEY']
    sentiment_tags = ['O', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG']

    processed = 0
    text_input = ''

    # example sentence
    examples = [
        'Sesuai diskusi. Barang mulus walau ada hairline scratch di casing depan. Battery masih awet. Dapat bonus keyboard Jepang jg. Thanks gan.',
        'product.. Ram 8GB HDD 320 sesuai pesanan. body luar dalam mulus, gores kecil pemakaian wajar.. keyboard empuk. display no dot Pixel. wifi. usb. sound. bluetooth. charger baterai ok awet 😁😁 over all sangat recommended mantaaaap jiwaaaaa',
        'overall, laptopnya compact, keyboard berfungsi semua, LED msh bagus, body bersih, charger berfungsi dgn baik, wifi jg connect, dan sudah terinstall win x. recommended',
        'Barang bagus, desain kokoh, keyboard empuk. Hanya saja mendapatkan laptop yg keyboard nya macet"an. Tp penjual sudah tanggung jawab.',
        'laptop sudah terima, tapi merk beda diterima Lenovo  tanpa kode  bukan Toshiba kode  731 732  dan baterainya tidak dapat nyimpan serta asesoreslain TDK ada seperti key bord, tas,dsb.',
        'terima kasih paket sudah diterima, hanya setelah dipakai ada beberapa Tut keyboardnya eror alias tidak bisa digunakan',
        'kurang memuaskan..batrenya gak kuat jangankan sehari 1jam aja gak kuat..sebelum di kirim harusnya di cek dulu..kalo udah gini gemanaaa..',
        'barang udah sampai dan saya terima,, tapi kibotya rusak sebahagian gak berpungsi dan anehya lagi baut sebahagian gak ada ini copot dari tokoya atau kenak copot,, ini gimana solusinya kak',
        'bagus sih CMN nih laptop dlmnya kotor bgt kipas dlm laptopnya pun berdebu bgt bisa rusak KLO g dibersihkan untuk seller tolong diperhatikan juga kualitas barangnya seblom di pack tolong di bongkar dan bersihkan dlmnya.',
        'Bagus, batre baru, kabel power baru..banyak bonusnya juga, sayang sekali ada retak dibagian pinggir yg dipaksakan dilem pakai lem korea'
    ]

    # ======================================================================== PAGE LAYOUT ========================================================================    
    st.markdown("<h1 style='text-align: center; margin:0; padding:0;'>Analisis Sentimen Berbasis Aspek</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; margin:0; padding:0;'>pada Ulasan Produk Elektronik Laptop</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; margin-top:0.5em; color:#DC3545;'>oleh Hikmawati Fajriah Ayu Wardana (NIM 1903510)</h5>", unsafe_allow_html=True)

    "\n\n"

    css = import_css('style.css')

    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        input_val = ''
        text_input = st.text_area(label="Text 📝", placeholder="Masukkan ulasan...", value=input_val, key='text')
            
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("Hapus", use_container_width=True, on_click=clear_input):
                processed = 0
        with col2b:
            if st.button("Prediksi", use_container_width=True, type="primary"):
                processed = 1

    with col2:
        if processed == 1:

            output_aspect = get_output_format(tokenizer, text_input, aspect_model, MAX_LENGTH, aspect_tags, 'a')
            output_sentiment = get_output_format(tokenizer, text_input, sentiment_model, MAX_LENGTH, sentiment_tags, 's')

            all_token = []
            all_aspect = []
            all_sentiment = []
            for i in output_aspect:
                res = i.split(' ')
                token = res[0]
                if len(res[1]) > 1:
                    label = res[1][-3:]
                else:
                    label = res[1]
                all_token.append(token)
                all_aspect.append(label)
            for i in output_sentiment:
                res = i.split(' ')
                if len(res[1]) > 1:
                    label = res[1][-3:]
                else:
                    label = res[1]
                all_sentiment.append(label)

            # st.write(all_token)
            # st.write(all_aspect)
            # st.write(all_sentiment)

            highlighted_aspect = highlight_token(all_token, all_aspect)
            highlighted_sentiment = highlight_token(all_token, all_sentiment)

            aspect_label = "<div class='container'><span class='label-font'>Aspek 💻</span></div>"
            div_aspect = f"<div class='output'>{aspect_label} {highlighted_aspect}</div>"
            st.markdown(div_aspect, unsafe_allow_html=True)

            sentiment_label = "<div class='container'><span class='label-font'>Sentimen 🎭</span></div>"
            div_sentiment = f"<div class='output2'>{sentiment_label} {highlighted_sentiment}</div>"
            st.markdown(div_sentiment, unsafe_allow_html=True)
        else:
            input_val = ''
            div = f"<div class='custom-div'></div>"

            aspect_label = "<div class='container'><span class='label-font'>Aspek 💻</span></div>"
            div_aspect = f"<div class='output'>{aspect_label} {div}</div>"
            st.markdown(div_aspect, unsafe_allow_html=True)

            sentiment_label = "<div class='container'><span class='label-font'>Sentimen 🎭</span></div>"
            div_sentiment = f"<div class='output2'>{sentiment_label} {div}</div>"
            st.markdown(div_sentiment, unsafe_allow_html=True)
    
    # examples:
    st.markdown("<div class='container3'><span class='label-font'>Contoh</span></div>", unsafe_allow_html=True)
    for i in examples:
        st.button(i, on_click=set_input, args=(i,))

if __name__ == "__main__":
    main()