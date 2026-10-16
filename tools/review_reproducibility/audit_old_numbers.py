# -*- coding: utf-8 -*-
import sys, io, docx
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
doc = docx.Document('c:/Users/cssp/Desktop/D_folder/tools/Tools_env/RTM/RTM_Paper_Final_Camera_Ready_v2.docx')

targets = ['8.3%', '1/12', '25.0%', '3/12', '0/10', '1.5/12', '0.5/10', '10.7', '5.0 ms', '5.0 ms', '0.957', '0.818', '0.700', '100×', '100x']

print('--- Scanning Paragraphs ---')
for i, p in enumerate(doc.paragraphs):
    for t in targets:
        if t in p.text:
            print(f'P[{i}] match "{t}": {p.text[:120]}...')

print('\n--- Scanning Tables ---')
for t_idx, tbl in enumerate(doc.tables):
    for r_idx, row in enumerate(tbl.rows):
        for c_idx, cell in enumerate(row.cells):
            for t in targets:
                if t in cell.text:
                    print(f'Table {t_idx} R{r_idx}C{c_idx} match "{t}": {cell.text}')
