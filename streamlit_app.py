import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

def format_linenum(line_idx, bounds, filenames):
    page_idx = np.where(line_idx <= bounds)[0][0]
    if page_idx > 0:
        line_idx -= bounds[page_idx-1]
    return page_idx, line_idx-1

def update_entities(entities, bounds, filenames, data):
    for d in data:
        try:
            if d['Тип'] not in entities:
                entities[d['Тип']] = {}
            for a in d['Упоминания']:
                a['page_idx'],a['line_idx'] = format_linenum(a['Номер строки'], bounds, filenames)
            if d['Имя'] not in entities[d['Тип']]:
                entities[d['Тип']][d['Имя']] = {'Описание': d['Описание'], 'Упоминания': d['Упоминания']}
            else:
                entities[d['Тип']][d['Имя']]['Описание'] += ' ' + d['Описание']
                entities[d['Тип']][d['Имя']]['Упоминания'] += d['Упоминания']
        except:
            pass
    return entities

def main():
    
    st.set_page_config(layout="wide")
    
    if 'filenames' not in st.session_state:
        filenames = os.listdir('labels')
        filenames = [fn for fn in filenames if fn.endswith('.txt')]
        filenames.sort(key=lambda x: 1000*int(x[6:8]) + (int(x[11:-4]) if x[11:-4].isdigit() else int(x[11:-6])) + 0.5*('об' in x))
        markup = []
        for fn in filenames:
            with open(f'labels/{fn}', encoding='utf-8') as f:
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
        files = sorted([fn for fn in files if fn.endswith('.json')])
        entities = {}
        for fn in files:
            with open(f'jsons/{fn}', 'rb') as json_data:
                data = json.load(json_data)
            entities = update_entities(entities, bounds, filenames, data)
        st.session_state['entities'] = entities
    else:
        entities = st.session_state['entities']    
    ent_type = st.sidebar.selectbox('Выберите тип сущности:', entities.keys())
    ent_name = st.sidebar.selectbox('Выберите сущность:', sorted(entities[ent_type].keys()))
    
    st.sidebar.text_area(label='Описание', value=entities[ent_type][ent_name]['Описание'], height=100, disabled=True, label_visibility="collapsed")
    mentions = entities[ent_type][ent_name]['Упоминания']
    values = [(('0' if m['Дата'][1] == ' ' else '') + m['Дата'] if 'Дата' in m else 'Неизвестно') + ' – ' + m['Роль'] for m in mentions]
    ent_case = st.sidebar.selectbox('Выберите упоминание:', values)
    
    ent_idx = values.index(ent_case)
    page_idx,line_idx = mentions[ent_idx]['page_idx'],mentions[ent_idx]['line_idx']
    
    json_fn = st.sidebar.file_uploader('Загрузить свой список', type=['.json','.txt'], accept_multiple_files=True)
    if 'json_names' not in st.session_state:
        st.session_state['json_names'] = []
        
    json_names = sorted([fn.name for fn in json_fn])
    if json_names != st.session_state['json_names']: 
        if json_names:
            entities = {}
            for fn in json_fn:
                data = json.load(fn)
                entities = update_entities(entities, bounds, filenames, data)
        else:
            files = os.listdir('jsons')
            files = sorted([fn for fn in files if fn.endswith('.json')])
            entities = {}
            for fn in files:
                with open(f'jsons/{fn}', 'rb') as json_data:
                    data = json.load(json_data)
                entities = update_entities(entities, bounds, filenames, data)
        st.session_state['entities'] = entities
        st.session_state['json_names'] = json_names
        st.rerun()
    
    st.markdown("## Навигация по дневникам А.В. Сухово-Кобылина")
    img = cv2.imdecode(np.fromfile(f"images/{filenames[page_idx][:-4] + '.jpg'}", dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    
    col1,col2 = st.columns([0.45,0.55])
    
    with col1:
        st.markdown(f'##### {filenames[page_idx][:-4]}, строка {line_idx+1}')
        step = 2
        text = ''.join(l + '  \n' for l in markup[page_idx]['lines'][max(line_idx-step,0):line_idx])
        text += '**' + markup[page_idx]['lines'][line_idx] + '**  \n'
        text += ''.join(l + '  \n' for l in markup[page_idx]['lines'][line_idx+1:line_idx+step+1])                                    
        st.markdown(text.replace('#', '<...>').replace('.','\.'))
        
    with col2:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_axis_off()
        ax.imshow(img[...,(2,1,0)])
        h,w = img.shape[0],img.shape[1]

        for i,box in enumerate(markup[page_idx]['boxes']):
            plt.plot(np.array(box)[[1,3,5,7,1]]*w, np.array(box)[[2,4,6,8,2]]*h, linewidth=0.5, color='r' if i == line_idx else 'b')
        st.pyplot(fig, use_container_width=False)
    
    
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass