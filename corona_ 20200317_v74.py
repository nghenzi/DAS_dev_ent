# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:06:27 2020

@author: 54911
"""

import numpy as np
import matplotlib
import tkinter as tk
import keyboard
import time
import collections
import os
import glob
import sys
import matplotlib.patches as ptc
import warnings

from itertools import groupby
from operator import itemgetter
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import Tk, TOP, BOTH, LEFT, Text, E, W, S, N, END
from tkinter import NORMAL, DISABLED, StringVar, Label
from tkinter import filedialog
from multiprocessing import Process, Manager, Queue
from multiprocessing import freeze_support, shared_memory
from queue import Empty
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import path
from sklearn.cluster import DBSCAN

from db_func_ndb import reporte_db, db_rep_th

matplotlib.use('TkAgg')
warnings.simplefilter('ignore',np.RankWarning)

desvios_path=os.getcwd()
### all this must be global to share between processes 
end_bin = 4000
bins_constant = 22201
str_geometry = "1050x650+100+10"

f = Figure(figsize=(8, 5.5), dpi=100)
ax = f.add_subplot(211)
ax.set_xlim([0, end_bin])
ax.set_ylim([1000, 0])
ax_bin = f.add_subplot(212)
#ax.set_title('Waterfall Data') 
b = np.random.random((1000,end_bin))
im = ax.imshow(b, aspect='auto', cmap='jet',interpolation=None, clim=[0,0.2])
im_bin = ax_bin.imshow(b, aspect='auto', cmap='binary_r', interpolation=None, clim=[0,0.2])
f.canvas.draw() ## maybe it´s not necessary 

q = Queue()
q_lett = Queue()

class dasGUI(tk.Tk):

    def __init__(self, *args, **kwargs):  #kwargs Dictionary, args - arguments\
#        global f, ax, ax_bin, im, im_bin
#        global shm 
        global q_lett, shm, shared_list, bordes, std_ini,cluster_report_dicc, report_dicc_circula 
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.title(self, "Development Environment DAS")
        tk.Tk.iconbitmap(self,default='sur_logotipo_tilde_vertical_743_icon.ico')
        
        
        
        container = tk.Frame(self)
#        container.config(bg="red")
#        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)


        menubar = tk.Menu(container)
        
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New Experiment", command= lambda: popupmsg("that is not defined yet"))
        filemenu.add_separator()
        filemenu.add_command(label="Run From a File", command= quit)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command= quit)
        menubar.add_cascade(label="File", menu=filemenu)
        
        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Load data folder", command= self.get_path_data)
        editmenu.add_separator()
        editmenu.add_command(label="Terminate process data quit", command= self.terminate)
        editmenu.add_separator()
        editmenu.add_command(label="Exit", command= quit)
        menubar.add_cascade(label="Data", menu=editmenu)
        
        runmenu = tk.Menu(menubar, tearoff=0)
        runmenu.add_command(label="New Experiment", command= lambda: popupmsg("that is not defined yet"))
        runmenu.add_separator()
        runmenu.add_command(label="Run From a File", command= quit)
        runmenu.add_separator()
        runmenu.add_command(label="Exit", command= quit)
        menubar.add_cascade(label="Run", menu=runmenu)
        
        windowmenu = tk.Menu(menubar, tearoff=0)
        windowmenu.add_command(label="New Experiment", command= lambda: popupmsg("that is not defined yet"))
        windowmenu.add_separator()
        windowmenu.add_command(label="Run From a File", command= quit)
        windowmenu.add_separator()
        windowmenu.add_command(label="Exit", command= quit)
        menubar.add_cascade(label="Window", menu=windowmenu)
        
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="New Experiment", 
                    command= lambda: popupmsg("that is not defined yet"))
        helpmenu.add_separator()
        helpmenu.add_command(label="Run From a File", command=quit)
        helpmenu.add_separator()
        helpmenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="Help", menu=helpmenu)
        
        tk.Tk.config(self, menu=menubar)
        
#        self.frames = {} #dicfrom matplotlib.figure import Figure
        self.button1 = ttk.Button(self, text="start", command=self.onStart)
        self.button2 = ttk.Button(self, text="forward", command=self.forward)
        self.button3 = ttk.Button(self, text="backward", command=self.backward)
        self.button4 = ttk.Button(self, text="pause", command= self.pause)
        self.button5 = ttk.Button(self, text="next", command=self.nexte)
        self.button6 = ttk.Button(self, text="previous", command=self.nexte)
        self.txt = scrolledtext.ScrolledText(self,  wrap   = tk.WORD, 
                                             width  = 20, height = 25)          
        self.L1 = ttk.Label(self, text="Bins")
        self.entry1 = ttk.Entry(self)
        self.L2 = ttk.Label(self, text="Sigma 1")
        self.entry2 = ttk.Entry(self)
        self.L3 = ttk.Label(self, text="Sigma 2")
        self.entry3 = ttk.Entry(self)
        self.L4 = ttk.Label(self, text="Sigma 3")
        self.entry4 = ttk.Entry(self)
        fr = tk.Frame(self)
        
        
        self.button1.grid(row=0, column=0, padx=5, pady=5, sticky=E+W+S+N)
        self.button2.grid(row=0, column=1, padx=5, pady=5, sticky=E+W+S+N)
        self.button3.grid(row=0, column=2, padx=5, pady=5, sticky=E+W+S+N)
        self.button4.grid(row=1, column=0, padx=5, pady=5, sticky=E+W+S+N)
        self.button5.grid(row=1, column=1, padx=5, pady=5, sticky=E+W+S+N)
        self.button6.grid(row=1, column=2, padx=5, pady=5,sticky=E+W+S+N)
        self.txt.grid(row=6, column=0, columnspan=3, rowspan=5, padx=5, pady=5)
        self.L1.grid(row=2,column=0)
        self.entry1.grid(row=2, column=1)
        self.L2.grid(row=3,column=0)
        self.entry2.grid(row=3, column=1)
        self.L3.grid(row=4,column=0)
        self.entry3.grid(row=4, column=1)
        self.L4.grid(row=5,column=0)
        self.entry4.grid(row=5, column=1)
        
        fr.grid(row=0, column=3, columnspan=3,rowspan=20, sticky=N+W+E+S)
        
             

        self.canvas = FigureCanvasTkAgg(f, master=fr)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        toolbar = NavigationToolbar2Tk(self.canvas, fr)
        toolbar.update()
        self.canvas._tkcanvas.pack()
        
        
        prompt = '      Press any key      '
        label1 = Label(self, text=prompt, width=len(prompt), bg='yellow')
        label1.grid(row=0, column=7)
        
        self.flagP=True
        
        def key(event):
            global shared_list, bordes, cluster_report_dicc, report_dicc_circula 
            if event.char == event.keysym:
                msg = 'Normal Key %r' % event.char
                if event.char=='k':
                    if self.flagP: 
                        start=time.time()
                        plot_binary(f,im, im_bin, self.b, self.b_bin, ax, ax_bin)
                        print  ('__plotting time:', time.time()-start)
            elif len(event.char) == 1:
                msg = 'Punctuation Key %r (%r)' % (event.keysym, event.char)
            else:
                msg = 'Special Key %r' % event.keysym
            label1.config(text=msg)
            
        self.bind_all('<Key>', key)
#        self.background = f.canvas.copy_from_bbox(ax.bbox)
#        print (im.get_array()[:10, 1])

    def terminate(self):
        global shm, shm_bin, shared_list, bordes, cluster_report_dicc, report_dicc_circula 
        self.p1.terminate()
        self.shm.close()
        self.shm_bin.close()
        self.shm.unlink()
        self.shm_bin.unlink()
#        shared_list.remove()
        
    def forward(self):
        q_lett.put('f')
        self.flagP= True 
    
    def backward(self):
        q_lett.put('b')
        self.flagP= True 
    
    def prev(self):
        q_lett.put('a')
        self.flagP= True 
    
    def nexte(self):
        q_lett.put('n')
        self.flagP= True
        
    def pause(self):
        q_lett.put('p')
        self.flagP= False 
        
    def get_path_data(self):
        global path_data
        path_data=filedialog.askdirectory()

    def onStart(self):
        global im, f,shm, shm_bin, path_data, q_lett, shared_list, bordes, std_ini
        global cluster_report_dicc, report_dicc_circula 
       
        a = np.random.random((1000,end_bin)) # Start with an existing NumPy array
        self.shm = shared_memory.SharedMemory(name='memo04_wf1460', create=True, size=a.nbytes)
        self.shm_bin = shared_memory.SharedMemory(name='memo04_wf1460_bin', create=True, size=a.nbytes)
        
        self.b = np.ndarray((1000,end_bin), dtype=np.float64, buffer=self.shm.buf)
        self.b_bin = np.ndarray((1000,end_bin), dtype=np.float64, buffer=self.shm_bin.buf)
        
        self.p1 = Process(target=cluster_proc, args=(q_lett,1, 1,path_data,shared_list, bordes, std_ini,
                                                     cluster_report_dicc, report_dicc_circula ))
        self.p1.start()
#        self.pbar.start(DELAY2)      
   
    
def plot_binary( f, im, im_bin, b, b_bin, ax, ax_bin):
        
        global shared_list, bordes, report_dicc_circula, cluster_report_dicc
        
        
        print ('la len de shared list is:', len(shared_list))
        try:
            print ('first SL element is:', shared_list[0])
#            pepe=shared_list.copy()            
        except:
            pass
        
        ### Borro data de la iteracion anterior 
        for t in ax.texts.copy(): t.remove()
        for line in ax.lines.copy(): line.remove()
        for pc in ax.patches.copy(): pc.remove() 
        
        for tup in shared_list[:]:
                ### unpacking tup 
                cluster,cs,not_cs,fit,ll,w,h,good_fit, edgecolor, \
                txt_c, fontsize,velocidad, txt_data,\
                delta_posicion, intensidad_posicion, delta_tiempo, intensidad_tiempo=tup
                
                if h>50 and good_fit:
                    print ('cluster us', cluster.shape, velocidad)
                    dada=ax.plot(cluster[:,0],cluster[:,1],'o')
#                    ax.draw_artist(dada) 
                
                rect_clust=ptc.Rectangle(ll,w,h,facecolor='none',edgecolor=edgecolor)
                ax.add_patch(rect_clust)
            
        for ann in bordes[:]: 
            vh, st, st_cant, txt_data= ann[0], ann[1], ann[2], ann[3]
            text_report=vh+'\n'+ st+':'+str(st_cant)+'\n'+ txt_data
            ax.annotate(text_report, ann[4], color=ann[5], fontsize=ann[6])
        
#pepe='str\ncopy\nfds'       
        
        print ('report',report_dicc_circula)
        print (' cluster report',cluster_report_dicc)
        im.set_data(b)
        im_bin.set_data(b_bin)

        f.canvas.draw()
    
#        ax.draw_artist(im)               
#        ax_bin.draw_artist(im_bin)
#        f.canvas.blit(ax.bbox)
#        f.canvas.blit(ax_bin.bbox)
        
        shared_list[:]=[]
        bordes[:]=[]
#        shared_list=[]
    
    
    
def cluster_proc(q_lett, digs, acc,path_data, shared_list, bordes, std_ini,
                 cluster_report_dicc, report_dicc_circula ):

    q_db = collections.deque()
    
    extendidos={}
    
    
    existing_shm = shared_memory.SharedMemory(name='memo04_wf1460')
    existing_shm_bin = shared_memory.SharedMemory(name='memo04_wf1460_bin')
    
    c = np.ndarray((1000,end_bin), dtype=np.float64, buffer=existing_shm.buf)
    c_bin = np.ndarray((1000,end_bin), dtype=np.float64, buffer=existing_shm_bin.buf)
    
#    q.put('a dta es')
    
    print ('salgo de process 2 to sleep:', c[-20:,-1])
#    du=collections.deque()
#    da=collections.deque()
#    std_ini=0
    bins= bins_constant
    try:
        bins = 22201
        big_wf = load_data( std_ini, bins,path_data)
    except:
        pass
    
    try:
        bins = 22500
        big_wf = load_data( std_ini, bins,path_data)
    except:
        pass
    
    try:
        bins = 14261
        big_wf = load_data( std_ini, bins,path_data)
    except:
        pass
        


    bin_filt_1 = 25
    bin_filt_2 = 15
    bin_filt_3 = 8
    posicion1 = 5500
    posicion2 = 8700
    
    desplazo = 10 
    Niter = int(big_wf.shape[0]/desplazo)
   
#    pi=len(du)
    
    print ('termine de cargar la data')
    os.chdir(desvios_path)
    mad_data = np.load("desvios.npy")
    median_data = np.load("medias.npy")
    base_fin = median_data.size
    
    c[:,:] = np.eye(N=1000,M=   end_bin) *0.2
        
    wf_actual = c.copy()
    wf_actual_bin = c_bin.copy()
    
    print ('antes de entrar al loop', wf_actual.shape)
    tiempo=0

    
    medianM = np.zeros((desplazo, end_bin))
    madM =   np.zeros((desplazo, end_bin)) 
    for jj in range(desplazo):
        medianM[jj,:] = median_data[:end_bin]
        madM[jj,:]= mad_data[:end_bin]
        
        
    flag_forward=True    
    flag_backward=False
    flag_next=False
    flag_previous=False
     
    jkl=-1
    while jkl < Niter:
#        jkl+=1
#        time.sleep(0.4)
        
        
        if q_lett.qsize()>0:
            print ('entro a la cola' )
            letter=q_lett.get_nowait()
            if letter=='f':
                flag_forward=True
                flag_backward=False
                flag_next=False
                flag_previous=False
            if letter=='b':
                flag_forward=False
                flag_backward=True
                flag_next=False
                flag_previous=False
#                hacer adelantar la data
            if letter=='n':
                flag_forward=False
                flag_backward=False
                flag_next=True
                flag_previous=False
#                hacer adelantar la data
            if letter=='a':
                flag_forward=True
                flag_backward=False
                flag_next=False
                flag_previous=True
#                hacer adelantar la data
            if letter=='p':
                flag_forward=False
                flag_backward=False
                flag_next=False
                flag_previous=False
#                hacer adelantar la data
        
        if not flag_forward and not flag_backward and not flag_next and not flag_previous:
            time.sleep(0.5)
            
        if flag_forward:
            jkl+=1
            tiempo=jkl
            wf_actual=np.roll(wf_actual,-desplazo,axis=0)
            wf_actual_bin=np.roll(wf_actual_bin,-desplazo,axis=0)
#                st3=time.time()
            wf_actual[-desplazo:,:]=big_wf[tiempo*desplazo:(tiempo+1)*desplazo,:]
            MAD=np.abs(big_wf[tiempo*desplazo:(tiempo+1)*desplazo,:]-medianM)/madM
            wf_actual_bin[-desplazo:,:posicion1]= MAD[:,:posicion1] > bin_filt_1
            wf_actual_bin[-desplazo:,posicion1:posicion2]= MAD[:,posicion1:posicion2] > bin_filt_2
            wf_actual_bin[-desplazo:,posicion2:]= MAD[:,posicion2:] > bin_filt_3
            
        if flag_backward:
            jkl-=1
            tiempo=jkl
            wf_actual=np.roll(wf_actual,desplazo,axis=0)
            wf_actual_bin=np.roll(wf_actual_bin,desplazo,axis=0)
#                st3=time.time()
            wf_actual[:desplazo,:]=big_wf[tiempo*desplazo:(tiempo+1)*desplazo,:]
            MAD=np.abs(big_wf[tiempo*desplazo:(tiempo+1)*desplazo,:]-medianM)/madM
            wf_actual_bin[:desplazo,:posicion1]= MAD[:,:posicion1] > bin_filt_1
            wf_actual_bin[:desplazo,posicion1:posicion2]= MAD[:,posicion1:posicion2] > bin_filt_2
            wf_actual_bin[:desplazo,posicion2:]= MAD[:,posicion2:] > bin_filt_3
        
        if flag_next:
            jkl+=1
            tiempo=jkl
            flag_next=False
            
        if flag_previous:
            jkl-=1
            tiempo=jkl
            flag_previous=False
            
            
        ### AVANZO TEMPORALMENTE SEA CUAL SEA EL FLAG DE EVOLUCION ... 
        ### aca entro a calcular clusters y propiedades porque modifique el tiempo 
        if flag_forward or flag_backward or flag_next or flag_previous:
            keyboard.press_and_release('k') ## with k letter, the other process executes plot_binary to update the mpl figure
#            k
             
            
            
            
            for key, group in groupby(sorted(list(extendidos.keys()), key=lambda x: x[0]), lambda x: x[0]):         
                for thing in group:
                    try:  
                        clave= max([item for item in group], key=itemgetter(1))
                        print ("grouped: %s : %s " % (key, clave))
                        
                        print (clave)
                        if min(extendidos[clave][0][:,1]) < desplazo:
                            cluster,cs,not_cs,fit,ll,w,h,good_fit, edgecolor, \
                            txt_c, fontsize,velocidad, txt_data,\
                            delta_posicion, intensidad_posicion, delta_tiempo, intensidad_tiempo= extendidos[clave]
                            
                            cluster[:,1] -= desplazo
            
                            extendidos[clave]= cluster,cs,not_cs,fit,ll,w,h,good_fit, edgecolor, \
                            txt_c, fontsize,velocidad, txt_data,\
                            delta_posicion, intensidad_posicion, delta_tiempo, intensidad_tiempo
                    except:
                        pass
            
            
            time_da=tiempo
            
            ctr=0
            threshold=0.03 
            #parametro cluster primera filtrada
            eps=4#15
            min_samples=21#250
            filas_imagen_binaria=1000
            size_wf=(filas_imagen_binaria,median_data.size)
                
            startu=time.time()
            pts_cluster=np.transpose(np.array(((np.nonzero(wf_actual_bin)[1], np.nonzero(wf_actual_bin)[0]))))
    
            if len(pts_cluster.shape)>1:
                db=DBSCAN(eps=eps    ,min_samples=  min_samples).fit(pts_cluster)
            else:
                time.sleep(0.01)
                
            finu= time.time()
                    
            rr, noise_pts= calculo_cluster_properties(db, pts_cluster,wf_actual)   
            
            gnu=time.time()
            
#            shared_list=[]
            ### extiendo clusters 
           
            flag_extend=True
            while flag_extend:
                rr,noise_pts, flag_extend= extiendo_clusters(rr, noise_pts, db, pts_cluster,wf_actual, cluster_report_dicc)        
                print (flag_extend) 
             
            ### chequeo clusters reportados extendidos y no cortados 
            ###tiempo es el tiempo de abajo
            ### tiempo - 1000 es el tiempo de arriba
            lista_posiciones= [int(item[4][0]) if item[11]>0 else int(item[4][0]+item[5]) for item in  rr]
            
            for p,n in list(cluster_report_dicc.keys())[::]  :
                
#                extendidos[p,n,] 
                
                if p in lista_posiciones: # print (p)
                    print (p)
                    idx_cluster_to_modify= np.where(np.array(lista_posiciones)==p)[0][0]
                    tup=rr[ idx_cluster_to_modify ]
                    cluster,cs,not_cs,fit,ll,w,h,good_fit, edgecolor, \
                    txt_c, fontsize,velocidad, txt_data,\
                    delta_posicion, intensidad_posicion, delta_tiempo, intensidad_tiempo=tup
#                    print (np.min(cluster[:,1]) , 'jjkl el cluster es;,', cluster[:,1])
#                    min_tiem=  
                    keys=( p, n, (tiempo+1)*desplazo - 1000 +  np.min(cluster[:,1]))
                    
                    if good_fit: extendidos[keys]=tup
#                        extendidos[keys]= cluster,cs,not_cs,fit,ll,w,h,good_fit, edgecolor, \
#                        txt_c, fontsize,velocidad, txt_data,\
#                        delta_posicion, intensidad_posicion, delta_tiempo, intensidad_tiempo=tup
                        
#                        for tup in rr:
#                        shared_list.append(tup)
            

            print ('')
            print ('')
            print('tiempo:',tiempo*desplazo)
#            for things in extendidos.keys():
            print ('kelsssssss', extendidos.keys())
            for key, group in groupby(sorted(list(extendidos.keys()), key=lambda x: x[0]), lambda x: x[0]):
                
               for thing in group:
                    try:  
                        clave= max([item for item in group], key=itemgetter(1))
                        print ("grouped: %s : %s " % (key, clave))
                        
                        print (clave)
                        if min(extendidos[clave][0][:,1]) < desplazo:
                            cluster,cs,not_cs,fit,ll,w,h,good_fit, edgecolor, \
                            txt_c, fontsize,velocidad, txt_data,\
                            delta_posicion, intensidad_posicion, delta_tiempo, intensidad_tiempo= extendidos[clave]
                                
                        
                        print ('limites time clust', min(extendidos[clave][0][:,1]), max(extendidos[clave][0][:,1]))
                
                        shared_list.append(extendidos[clave])
                        
                    except:
                        pass
            print ('')
            print ('')
#            time.sleep(1)
            
            

            
            print ('updating shm c, ___ iteracion n:', jkl ) ###(jkl* desplazo is the time executed... )
            c[:,:]= wf_actual.copy()    
            c_bin[:,:]= wf_actual_bin.copy()
            
            
            print ('DBSCAN time is:', finu-startu)
            print ('cluster_ properties time is:', gnu-finu)

            
##            rr, noise_pts= extiendo_clusters(rr, noise_pts, db, pts_cluster,wf_actual, cluster_report_dicc,ax)
#            ### aca itero sobre todos los clusters, esto es el lugar donde tengo que extenderlos. 
#            ### en cluster fit.... 
##            for cluster,cs,not_cs,fit,ll,w,h,good_fit,edgecolor,txt_c,fontsize,velocidad, txt_data in rr:
##%%
#            bordea=[]
                ### aca solo clasifico segun las caracteristicas ,,,,    
            for tup in rr:
#                ### unpacking tup 
                cluster,cs,not_cs,fit,ll,w,h,good_fit, edgecolor, \
                txt_c, fontsize,velocidad, txt_data,\
                delta_posicion, intensidad_posicion, delta_tiempo, intensidad_tiempo=tup
#                jom+=1
#                rect_clust=ptc.Rectangle(ll,w,h,facecolor='none',edgecolor=edgecolor)
#                ax.add_patch(rect_clust)
#                rect_clust_bin=ptc.Rectangle(ll,w,h,facecolor='none',edgecolor=edgecolor)
##                ax_bin.add_patch(rect_clust_bin)
#                bordes.append(rect_clust)
##                ax.plot(cluster[:,0],cluster[:,1],'o')
#                tr=(ll[0]+w,ll[1] + h)
#                ax.annotate(delta_tiempo,tr)
#                    
                if h>49 and 2000>w>10:
#                    rect_clust=ptc.Rectangle(ll,w,h,facecolor='none',edgecolor=edgecolor)
#                    ax.add_patch(rect_clust)
#                    rect_clust_bin=ptc.Rectangle(ll,w,h,facecolor='none',edgecolor=edgecolor)
#                    ax_bin.add_patch(rect_clust_bin)
#                    bordes.append(rect_clust)
                    
                    tr=(ll[0]+w,ll[1]+h)
#                    ax.annotate(delta_posicion,tr)
                    pts_cant=cluster.size
                    
#                    ax.plot()
                    ### por ahora vamos a extender solo los clusters grandes
                    ### capaz que hay que extenderlos en calculo_cluster_properties
#                    yoffset=[0, 10,20,30,40]
                    
#                    if good_fit:
#                        for off in yoffset:
#                            w_clus= max(cluster[:,0]) - min(cluster[:,0])
#                            if velocidad<0: xfit= np.arange(min(cluster[:,0])- w ,max(cluster[:,0]))
#                            if velocidad>0: xfit= np.arange(min(cluster[:,0]), max(cluster[:,0]) + w)                 
#                            yfit= fit(xfit)
#                            ax.plot(xfit,yfit+off)
#                            ax_bin.plot(xfit,yfit+off)
#                            if velocidad<0:
#                                ax.add_patch(ptc.Rectangle(ll-(5,-h),10,off,facecolor='none',
#                                                           edgecolor=edgecolor))
##                                ax_bin.add_patch(ptc.Rectangle(ll-(5,-h),10,off,facecolor='none',edgecolor=edgecolor))
#                            if velocidad>0:
#                                ax.add_patch(ptc.Rectangle(tr+(5,0)),10,off,facecolor='none',
#                                             edgecolor=edgecolor)
#                                ax_bin.add_patch(ptc.Rectangle(tr+(5,0)),10,off,facecolor='none',edgecolor=edgecolor)
                            
                            
                            
                    
                    ## reporte db da el vehiculo
                    ## cluster_report ve si es actualizacion o finalizacion 
                    ## st tipo de actualizacion, st_cant cantidad de actualizacion 
                    if velocidad>0:
                        st, st_cant= cluster_report(velocidad,pts_cant,ll,h,cluster_report_dicc)
                        vh=reporte_db(st,st_cant,velocidad,good_fit,ll,h,w,q_db,report_dicc_circula,time_da)
                    if velocidad<0:
                        st, st_cant=cluster_report(velocidad,pts_cant,(ll[0]+w,ll[1]),h,cluster_report_dicc)
                        vh=reporte_db(st,st_cant,velocidad,good_fit,(ll[0]+w,ll[1]),h,w,q_db,report_dicc_circula,time_da)
                    if velocidad==0:
                        vh='nny'
                        st='none'
                        st_cant='-3'            

                    if (edgecolor=='r' or edgecolor=='g') and velocidad!=0:
                        
                        ann=(vh, st, st_cant, txt_data, tr, txt_c, fontsize)
                        bordes.append(ann)
                        
                        texto_dic=str( vh+'\n'+ st +':'+str(st_cant)+'\n'+ str(txt_data) )
                        texto_dic={'vehiculo':str(vh), 'st':st, 'st_cant':st_cant, 'w':str(int(w)),
                                   'h':int(h), 'lower':ll[0], 'left':ll[1], 'velocidad':str(round(velocidad,3)),
                                   'ajuste':str(round((h/w)/fit[1],3))}
                        
                         
                        
#                        texto_dic={'t':'t'}
#                        ann_vehi_st.append(texto_dic)
                        
#            bordes[:]=bordea[:]           
                        
                        #%%%%
def define_polygons(cluster,fit,velocidad,h,w):
    ### pg1 
    w=2.4*w
    if velocidad<0:
        bp1= (min(cluster[:,0])+ 10   , max(cluster[:,1])-60)
        bp2= (min(cluster[:,0])- w    , fit(min(cluster[:,0])- w) - 60)
        bp3= (min(cluster[:,0])- w+10 , fit(min(cluster[:,0])- w+10) + 10)
        bp4= (min(cluster[:,0])+ 10   , max(cluster[:,1]) + 10)
        polygon1=np.array(([bp1,bp2,bp3,bp4]))
        ### --- ###
#        pl=ax.plot(cp[0],cp[1],'Pk',markersize=24)
        cp1= (min(cluster[:,0]), max(cluster[:,1])+10)
        cp2= (min(cluster[:,0])- w   , fit(min(cluster[:,0])- w)+10)
        cp3= (min(cluster[:,0])- w+10, fit(min(cluster[:,0])- w+10)+20)
        cp4= (min(cluster[:,0])+10   , max(cluster[:,1])+20)
        cp5= (min(cluster[:,0])+10   , max(cluster[:,1])-10) 
        cp6= (min(cluster[:,0])      , max(cluster[:,1])-10)
        polygon2=np.array(([cp1,cp2,cp3,cp4,cp5,cp6]))
        ### --- ###
        dp1= (min(cluster[:,0]), max(cluster[:,1])+20)
        dp2= (min(cluster[:,0])- w   , fit(min(cluster[:,0])- w)+20)
        dp3= (min(cluster[:,0])- w+10, fit(min(cluster[:,0])- w+10)+30)
        dp4= (min(cluster[:,0])+10   , max(cluster[:,1])+30)
        dp5= (min(cluster[:,0])+10   , max(cluster[:,1])-10) 
        dp6= (min(cluster[:,0])      , max(cluster[:,1])-10)
        polygon3=np.array(([dp1,dp2,dp3,dp4,dp5,dp6]))
        ### --- ###
        ep1= (min(cluster[:,0]), max(cluster[:,1])+0)
        ep2= (min(cluster[:,0])- w   , fit(min(cluster[:,0])- w)+0)
        ep3= (min(cluster[:,0])- w+10, fit(min(cluster[:,0])- w+10)+10)
        ep4= (min(cluster[:,0])+10   , max(cluster[:,1])+10)
        ep5= (min(cluster[:,0])+10   , max(cluster[:,1])-10) 
        ep6= (min(cluster[:,0])      , max(cluster[:,1])-10)
        polygon4=np.array(([ep1,ep2,ep3,ep4,ep5,ep6]))
    
    
    ###################------------------- ###################################3
    
    if velocidad>0:
        bp1= (max(cluster[:,0])- 10   , max(cluster[:,1])-40)
        bp2= (max(cluster[:,0])+ w    , fit(max(cluster[:,0])+ w) - 40)
        bp3= (max(cluster[:,0])+ w-10 , fit(max(cluster[:,0])+ w-10) + 10)
        bp4= (max(cluster[:,0])- 10   , max(cluster[:,1]) + 10)
        polygon1=np.array(([bp1,bp2,bp3,bp4]))
        ### --- ###
#        cp=cp6
#        for cp in [bp1,bp2,bp3,bp4]: pl=ax.plot(cp[0],cp[1],'Pr',markersize=24)
        cp1= (max(cluster[:,0]), max(cluster[:,1])+30)
        cp3= (max(cluster[:,0])+ w   , fit(max(cluster[:,0])+ w)+10)
        cp2= (max(cluster[:,0])+ w-10, fit(max(cluster[:,0])+ w+10)+30)
        cp4= (max(cluster[:,0])+10   , max(cluster[:,1])+20)
        cp5= (max(cluster[:,0])+10   , max(cluster[:,1])-10) 
        cp6= (max(cluster[:,0])      , max(cluster[:,1])-10)
        polygon2=np.array(([cp1,cp2,cp3,cp4,cp5,cp6]))
#for cp in [cp1,cp2,cp3,cp4,cp5,cp6]: pl=ax.plot(cp[0],cp[1],'Pr',markersize=24)
        ### --- ###
        dp1= (max(cluster[:,0]), max(cluster[:,1])+40)
        dp3= (max(cluster[:,0])+ w   , fit(max(cluster[:,0])+ w)+20)
        dp2= (max(cluster[:,0])+ w-10, fit(max(cluster[:,0])+ w+10)+40)
        dp4= (max(cluster[:,0])+10   , max(cluster[:,1])+30)
        dp5= (max(cluster[:,0])+10   , max(cluster[:,1])-10) 
        dp6= (max(cluster[:,0])      , max(cluster[:,1])-10)
        polygon3=np.array(([dp1,dp2,dp3,dp4,dp5,dp6]))
#        for cp in [dp1,dp2,dp3,dp4,dp5,dp6]: pl=ax.plot(cp[0],cp[1],'Pk',markersize=24)
        ### --- ###
        ep1= (max(cluster[:,0]), max(cluster[:,1])+50)
        ep3= (max(cluster[:,0])+ w   , fit(max(cluster[:,0])+ w)+30)
        ep2= (max(cluster[:,0])+ w-10, fit(max(cluster[:,0])+ w+10)+50)
        ep4= (max(cluster[:,0])+10   , max(cluster[:,1])+40)
        ep5= (max(cluster[:,0])+10   , max(cluster[:,1])-10) 
        ep6= (max(cluster[:,0])      , max(cluster[:,1])-10)
        polygon4=np.array(([ep1,ep2,ep3,ep4,ep5,ep6]))
#        for cp in [ep1,ep2,ep3,ep4,ep5,ep6]: pl=ax.plot(cp[0],cp[1],'Pk',markersize=24)
        
    return polygon1, polygon2, polygon3, polygon4   



def extiendo_clusters(rr, noise_pts, db, pts_cluster,wf_actual, cluster_report_dicc):
    """
    see  https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    """
    flag_extend=False
    
    for p,n in list(cluster_report_dicc.keys())[::]  :  #print (p,n) #recorro los clusters importantes 
        ## identifico el cluster correspondiente. 
        lista_posiciones= [int(item[4][0]) if item[11]>0 else int(item[4][0]+item[5]) for item in  rr]
        
        if p in lista_posiciones: # print (p)
            print (p)
            idx_cluster_to_modify=np.where(np.array(lista_posiciones)==p)[0][0]
            tup=rr[ idx_cluster_to_modify ]
            cluster,cs,not_cs,fit,ll,w,h,good_fit, edgecolor, \
            txt_c, fontsize,velocidad, txt_data,\
            delta_posicion, intensidad_posicion, delta_tiempo, intensidad_tiempo=tup
            
            if good_fit: 
                xfit= np.arange(min(cluster[:,0])- w ,max(cluster[:,0]))
                off=20
                
#                ax.text(min(cluster[:,0]), max(cluster[:,1]),str(fit))
                polygon1,polygon2, polygon3, polygon4 = define_polygons(cluster,fit,velocidad,h,w)
                
                path1 = path.Path(polygon1)
                path2 = path.Path(polygon2)
                path3 = path.Path(polygon3)
                path4 = path.Path(polygon4)
#                patch1 = ptc.PathPatch(path1, facecolor='y', alpha=0.4)
#                patch2 = ptc.PathPatch(path2, facecolor='y', alpha=0.4)
#                patch3 = ptc.PathPatch(path3, facecolor='y', alpha=0.4)
#                patch4 = ptc.PathPatch(path4, facecolor='y', alpha=0.4)
#                ax.add_patch(patch1)
#                ax.add_patch(patch2)
#                ax.add_patch(patch3)
#                ax.add_patch(patch4)
#                
                modify_idx_rr= []
                
                ### here it adds other detected clusters
                for jkf, tap in enumerate(rr):
#                  if tap != tup: 
                  if jkf != idx_cluster_to_modify:  
                    Acluster,Acs,Anot_cs,Afit,All,Aw,Ah,Agood_fit, Aedgecolor, \
                    Atxt_c, Afontsize,Avelocidad, Atxt_data,\
                    Adelta_posicion, Aintensidad_posicion, Adelta_tiempo, Aintensidad_tiempo=tap
                    
                    if All[0] <1500 or All[0]>2000 and np.sign(velocidad)==np.sign(Avelocidad): 
                        for pathe in [path1,path2,path3,path4]:
                            idx_inside= pathe.contains_points(Acluster)
                            pts_inside=(Acluster[idx_inside,0],Acluster[idx_inside,1])
                            
                            if pts_inside[0].shape[0] > 5:
    #                            print (All)
                                cluster=np.append(cluster,Acluster,axis=0)
                                modify_idx_rr.append(jkf)
                                print ('idx:',jkf, 'len rr:', len(rr))
                                flag_extend=True
    
                #### here it adds noise pts, when there is more than 25 noise pts. 
                for pathe in [path1,path2,path3,path4]:
                        idx_inside= pathe.contains_points(noise_pts)
                        pts_inside=(noise_pts[idx_inside,0],noise_pts[idx_inside,1])
                
                        if pts_inside[0].shape[0] > 5:
#                            print (All)
                            cluster=np.append(cluster,noise_pts[idx_inside, :],axis=0)
                            a1= noise_pts
                            a2= noise_pts[idx_inside,:]       
                            a1_rows = set(map(tuple, a1))
                            a2_rows = set(map(tuple, a2))
                            noise_pts=np.array(list(a1_rows.difference(a2_rows)))
                            flag_extend=True
#                            modify_idx_rr.append(jkf)
#                            print ('idx:',jkf, 'len rr:', len(rr))
                      
                        
                 ####       
                            
                if len(modify_idx_rr)>0:
                    modify_idx_rr= set(modify_idx_rr)
                    print (modify_idx_rr, 'rr',len(rr))
                    for idx in reversed(sorted(modify_idx_rr)): del rr[idx]
                    
#                    plt.figure(); plt.plot(noise_pts[:,0],noise_pts[:,1],'o', markersize= 6, mfc='none', mec='y')
#                    ax.plot(cluster[:,0],cluster[:,1],'o', markersize= 6, mfc='none', mec='y')
                    
                    rr[ idx_cluster_to_modify ]= cluster,cs,not_cs,fit,ll,w,h,good_fit, edgecolor, \
                                     txt_c, fontsize,velocidad, txt_data,\
                                     delta_posicion, intensidad_posicion, delta_tiempo, intensidad_tiempo
                ## aca falta modificar los valores de rr, pts_cant, etc... 
                
#            if velocidad>0 and good_fit: 
#                xfit= np.arange(min(cluster[:,0]), max(cluster[:,0]) + w)                 
#            
#            try:
#                yfit= fit(xfit)
#            except:
#                pass
            
#            polygon=np.array(([ [2761,83],[2761,103],[2710,136],[2757,136],[2796,104],[2796,83]     ]))
#            path1 = path.Path(polygon1)
#            path2 = path.Path(polygon2)
#            patch1 = ptc.PathPatch(path1, facecolor='r', alpha=0.4)
#            patch2 = ptc.PathPatch(path2, facecolor='r', alpha=0.4)
#            ax.add_patch(patch1)
#            ax.add_patch(patch2)
#            
    return rr, noise_pts, flag_extend  

                        
                        
def cluster_report(vel,pts_cant,ll,h,report): ## report es cluster_report_dicc
    
    if ll[1]<5: ## para eliminar los casos largos 
        
        if h>=500:#casos largos
            return 'update',-1
        
        ## report es donde estan los clusters reportados 
        if (ll[0],pts_cant) in report and h<500:
#            report[ll[0],1e9)]=1
            
            #print 'end'
            if report[(ll[0],pts_cant)][1]>5:
                del report[(ll[0],pts_cant)] 
                return 'fin',-1
            else: 
                
                report[(ll[0],pts_cant)][1] += 1
                return 'update',report[(ll[0],pts_cant)][0]
            
        return 'discard',-2

                
    if (ll[0],pts_cant) in report: ## fin es cuando la cantidad de puntos no aumenta
        report[(ll[0],pts_cant)][0] += 1   ### fijo cantidad de actualizaciones  fin
        st_cant=report[(ll[0],pts_cant)][0]
#        del report[(ll[0],pts_cant)]
        
        if report[(ll[0],pts_cant)][1]>5:
            return 'fin',st_cant
        else: 
            
            report[(ll[0],pts_cant)][1] += 1
            return 'update',st_cant   
#        return 'fin',st_cant
    
    
    for ll_k,pts_cant_k in report.keys():  ## aca itero sobre los clusters anteriores 
        
        if ll_k==ll[0] and pts_cant_k < pts_cant: ## update porque pts_cant se actualizo 
            report[(ll[0],pts_cant)]= [report[(ll_k,pts_cant_k)][0]+1, 0]  ### fijo cantidad de actualizaciones  update
            del report[(ll_k,pts_cant_k)]
            return 'update',report[(ll[0],pts_cant)][0]
        
    report[(ll[0],pts_cant)]= [1,0] ### fijo cantidad de actualizaciones  inicio
    
    return 'inicio',report[(ll[0],pts_cant)][0]        

                
#    q.put('a dta es')  
def witx(c2,wf_cut):

#plt.plot(c2[:,1],c2[:,0],'o')
    
    delta_x=[]
    i_x=[]
    for i in range(int(min(c2[:,1]))+2, int(max(c2[:,1])), 1): ## vario tiempo , calculo delta x 
#        print (i) 
        x=c2[:,0]
        f=c2[:,1]
        g= np.array([i for _ in range(f.shape[0])])
        idx = np.argwhere(np.diff(np.sign(f - g)))
    
        ddx=  max(x[idx]) - min(x[idx])
        i_x.append( np.sum( wf_cut[i, int(min(x[idx])) : int(max(x[idx]))] ) ) 
        delta_x.append(ddx)
    delta_posicion=np.mean(delta_x)
    intensidad_posicion=np.mean(i_x)

    delta_t=[]
    i_t=[]
    for i in range(int(min(c2[:,0]))+2, int(max(c2[:,0])), 1): ## vario posicion , calculo delta t 
#        print i 
        t=c2[:,1]
        f=c2[:,0]
        g= np.array([i for _ in range(f.shape[0])])
        idx = np.argwhere(np.diff(np.sign(f - g)))
    
        ddt=  max(t[idx]) - min(t[idx])
        i_t.append( np.sum( wf_cut[int(min(t[idx])) : int(max(t[idx])), i ] ) ) 
        delta_t.append(ddt)
    delta_tiempo=np.mean(delta_t)
    intensidad_tiempo=np.mean(i_t)
    
    return delta_posicion, intensidad_posicion, delta_tiempo, intensidad_tiempo


def calculo_cluster_properties(db, pts_cluster, wf_actual):
    rre=[]
    rr,noise_pts=cluster_fit(db,pts_cluster)
      
    for cluster,cs,not_cs,fit in rr:
        
        ll,w,h=cluster_box(cluster)                
         

        wf_cut=wf_actual
        delta_posicion, intensidad_posicion, delta_tiempo, intensidad_tiempo= witx(cluster,wf_actual)

            
        if not(np.isclose(fit[1],0)):
            good_fit= 0.6<=np.abs((h/w)/fit[1])<=1.4
#            print ('cluster numero', jom-1)
            
            if 2000<ll[0]<3000: print (cluster.size, ll)
        else:
            good_fit=False

        ### si fitea bien es verde, si fitea mal es rojo...
        if good_fit:#posible error de div por zeroc
            edgecolor='g'
            txt_c='r'
            fontsize=22
        else:
            edgecolor='r'
            txt_c='gray'
            fontsize=18
            
        if not(np.isclose(fit[1],0)):
                        velocidad=4.06*3.6/fit[1]
                        txt_data=' '.join(['x:',str(int(w)),'y:',str(int(h)),'\nLL:',str(ll),'\n km/h:',str(round(velocidad,3)),'\n ajuste:',str(round((h/w)/fit[1],3))])
        else:
            velocidad=0
            txt_data='NO DATA'
            
        tup=(cluster,cs,not_cs,fit,ll,w,h,good_fit, edgecolor, 
             txt_c, fontsize,velocidad, txt_data,
             delta_posicion, intensidad_posicion, delta_tiempo, intensidad_tiempo) 
        rre.append(tup)
#    , good_fit, edgecolor, txt_c, fontsize
            
    return rre, noise_pts

def cluster_box(cluster):

    lower_left=np.array([np.min(cluster[:,0]),np.min(cluster[:,1])])

    width=np.linalg.norm(lower_left-np.array([np.max(cluster[:,0]),lower_left[1]]))

    height=np.linalg.norm(lower_left-np.array([lower_left[0],np.max(cluster[:,1])]))


    return lower_left, width, height
 

                       
def cluster_fit(db,pts):

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_

    unique_labels = set(labels)

    clusters=[]

    noise_pts=pts[labels == -1]

    

    for k in unique_labels:

        if k!=-1:

            class_member_mask = (labels == k)

            

            x=pts[class_member_mask & core_samples_mask]

#                en x va la posicion, en y va el tiempo
            ajuste=np.poly1d(np.polyfit(x[:,0], x[:,1], 1))  

            xy = pts[class_member_mask & core_samples_mask]

            cs_cnt=len(xy)

            xy = pts[class_member_mask & ~core_samples_mask]

            not_cs_cnt=len(xy)

            clusters.append((pts[class_member_mask],cs_cnt,not_cs_cnt,ajuste))

    return clusters,noise_pts 

                        

def load_data( std_ini, bins, path_data):
    i=0
    du=[2220,]
    cwd=os.getcwd()
#    global path_data
    os.chdir(path_data)
    std_files=sorted(glob.glob('0*.std'))
    longitud=len(std_files)
    
    
    count=0
    avg_files=sorted(glob.glob('../AVG/0*.avg'))
    if longitud>20:
        std_files=std_files[0:20]
        avg_files=avg_files[0:20]
    big_wf=np.zeros((1000*longitud, end_bin))
    for f_std,f_avg in zip(std_files,avg_files):
        print ('loading data', 100.* count/longitud, '____  la longitud es: ', longitud )
        
        current_std=np.fromfile(f_std,dtype=np.float32).reshape(-1,bins)
        current_avg=np.fromfile(f_avg,dtype=np.float32).reshape(-1,bins)
        current_wf=current_std/current_avg
   
        big_wf[count*1000:(count+1)*1000,:end_bin]=current_wf[:,:end_bin]
        count+=1
#        for row in current_wf:
#            
#            row=row[:end_bin]
##            du.append((row,f_std+'_'+str(i)))
##            du.append(row)
#            big_wf[i,:end_bin]=row
#            i=i+1
    print ('data loaded', len(du) )
    return big_wf
    

def forward(da,du):
#    if len(du)>0:
        da.append(du.popleft())
        # print ('da', len(da))
    # print ('du', du)
        return da,du

def backward(da,du):
#    if len(da)>0:
        du.appendleft(da.pop())
        # print ('da', len(da))
    # print ('du', du)
        return da,du


def update_image(image,new_rows):
    prev_data=image.get_array()
    if len(new_rows.shape)>1:
        rows_to_update=new_rows.shape[0]
    else:
        rows_to_update=1
    # new_data=np.append(prev_data[rows_to_update:,:],new_rows.reshape(rows_to_update,int(new_rows.size/rows_to_update)),axis=0)
    new_data=np.append(prev_data[rows_to_update:,:],new_rows,axis=0)
    image.set_data(new_data)
              
    
    
def update_image_backwards(image,new_rows):
    prev_data=image.get_array()
    if len(new_rows.shape)>1:
        rows_to_update=new_rows.shape[0]
    else:
        rows_to_update=1
    # new_data=np.append(prev_data[rows_to_update:,:],new_rows.reshape(rows_to_update,int(new_rows.size/rows_to_update)),axis=0)
    # new_data=np.append(new_rows.reshape(rows_to_update,int(new_rows.size/rows_to_update)),prev_data[:-rows_to_update,:],axis=0)
    new_data=np.append(new_rows,prev_data[:-rows_to_update,:],axis=0)
    image.set_data(new_data)
    
    
    
def main():
    freeze_support()
    global shm,path_data, shm_bin, std_ini  
    global shared_list, bordes
    global cluster_report_dicc, report_dicc_circula 
    
    try:
        if len(sys.argv)>0:
             std_ini=int(sys.argv[1])
        else:
             std_ini=0
    except:
        std_ini=0
        
        
        
    manager = Manager()
    shared_list = manager.list()
    bordes=manager.list()
    cluster_report_dicc= manager.dict() # reporte de clusters, ver que onda. 
    report_dicc_circula=manager.dict()    
    
    
    path_data= 'C:\\19_19_09_09_03_31\\STD/' 
    app = dasGUI()
    app.geometry(str_geometry)
    app.mainloop()  


if __name__ == '__main__':
    main() 