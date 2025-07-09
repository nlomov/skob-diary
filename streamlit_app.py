import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

def format_linenum(line_idx, bounds, filenames):
    img_idx = np.where(line_idx <= bounds)[0][0]
    if img_idx > 0:
        line_idx -= bounds[img_idx-1]
    return filenames[img_idx][:-4] + ', ' + str(line_idx)

def main():
    
    st.set_page_config(layout="wide")
    
    if 'filenames' not in st.session_state:
        filenames = os.listdir('labels')
        filenames = [fn for fn in filenames if fn.endswith('.txt')]
        filenames.sort(key=lambda x: 1000*int(x[6:8]) + (int(x[11:-4]) if x[11:-4].isdigit() else int(x[11:-6])) + 0.5*('об' in x))
        markup = []
        for fn in filenames:
            with open(f'labels/{fn[:-4]}.txt', encoding='utf-8') as f:
                text = f.read().strip().split('\n')
                markup.append({'filename': fn, 'lines': text[1::3], 'boxes': [list(map(float, t.split())) for t in text[0::3]]})
        
        bounds = np.cumsum(np.array([len(m['lines']) for m in markup]))
        st.session_state['filenames'] = filenames
        st.session_state['markup'] = markup
        st.session_state['bounds'] = bounds
    else:
        filenames = st.session_state['filenames']
        markup = st.session_state['markup']
        bounds = st.session_state['bounds']
    
    if 'entities' not in st.session_state:
        files = os.listdir('jsons')
        files = [fn for fn in files if fn.endswith('.json')]
        entities = {}
        for fn in files:
            with open(f'jsons/{fn}', 'rb') as json_data:
                data = json.load(json_data)
            for d in data:
                if d['Тип'] not in entities:
                    entities[d['Тип']] = {}
                mentions = [format_linenum(a['Номер строки'], bounds, filenames)  + ' - ' +a['Роль'] for a in d['Упоминания']]
                if d['Имя'] not in entities[d['Тип']]:
                    entities[d['Тип']][d['Имя']] = {'Описание': d['Описание'], 'Упоминания': mentions}
                else:
                    entities[d['Тип']][d['Имя']]['Описание'] += ' ' + d['Описание']
                    entities[d['Тип']][d['Имя']]['Упоминания'] += mentions
        st.session_state['entities'] = entities
    else:
        entities = st.session_state['entities']
    
    ent_type = st.sidebar.selectbox('Выберите тип сущности:', entities.keys())
    ent_name = st.sidebar.selectbox('Выберите сущность:', sorted(entities[ent_type].keys()))
    
    st.sidebar.text_area(label='Описание', value=entities[ent_type][ent_name]['Описание'], height=100, disabled=True, label_visibility="collapsed")
    
    ent_case = st.sidebar.selectbox('Выберите упоминание:', entities[ent_type][ent_name]['Упоминания'])
    
    st.markdown("## Навигация по дневникам А.В. Сухово-Кобылина")
    
    line_idx = int(ent_case.split()[2]) - 1
    img_idx = filenames.index(ent_case.split(',')[0] + '.txt')
    img = cv2.imdecode(np.fromfile(f'images/{filenames[img_idx][:-4]}.jpg', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    
    col1,col2 = st.columns([0.45,0.55])
    
    with col1:
        step = 2
        text = ''.join(l + '  \n' for l in markup[img_idx]['lines'][max(line_idx-step,0):line_idx])
        text += '**' + markup[img_idx]['lines'][line_idx] + '**  \n'
        text += ''.join(l + '  \n' for l in markup[img_idx]['lines'][line_idx+1:line_idx+step+1])                                    
        st.markdown(text.replace('#', '<...>').replace('.','\.'))
        
    with col2:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_axis_off()
        ax.imshow(img[...,(2,1,0)])
        h,w = img.shape[0],img.shape[1]

        for i,box in enumerate(markup[img_idx]['boxes']):
            plt.plot(np.array(box)[[1,3,5,7,1]]*w, np.array(box)[[2,4,6,8,2]]*h, linewidth=0.5, color='r' if i == line_idx else 'b')
        st.pyplot(fig, use_container_width=False)
    
    
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass