#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 02:10:00 2019

@author: kreetus
"""

import numpy as np
import time,datetime,pyodbc

def bin2prog(b):
    ref_342=((b*2.03)-4232)/1000.
    
    return 342+ref_342


def handle_report_dicc_from_cluster(ll,db_dicc,st,time_da): #dicc como mucho toma eje_x keys
    '''
    db_dicc is report_dicc_circula
    '''    
    
    key=ll[0]
    
    evento_reportado=(key in db_dicc) and (db_dicc[key]['reportado'])
    
    evento_no_reportado= (key not in db_dicc) or not(db_dicc[key]['reportado'])
    
    if st=='inicio':
        db_dicc[key]={'reportado':True,'id':int(time.time()),'time_da':time_da}
        return st,db_dicc[key]['id']
        
        #report ini,id++
    
    if st=='update':
        
        if evento_no_reportado:
            db_dicc[key]={'reportado':True,'id':int(time.time())}
            return 'inicio',db_dicc[key]['id'] #lo identifico a la mitad del cluster, reporta el inicio en un update del cluster
        else:
            if time.time()-db_dicc[key]['id']<(7*60):#si fue reportado hace poco, criterio independiente del cluster, verificar. maybe usar h o identificadores del cluster al registrar el inicio en db_dicc
                return st,db_dicc[key]['id']
            else:
                db_dicc[key]={'reportado':True,'id':int(time.time())}
                return 'inicio',db_dicc[key]['id']
        #if evento_reportado:
                    
        
    if st=='fin':
        
        if evento_reportado:
            id_rep=db_dicc[key]['id']
            
            db_dicc[key]['reportado']=False
            db_dicc[key]['id']=None
            
            return st, id_rep


def reporte_db(st,st_cant,velocidad,good_fit,ll,h,w,q_db,report_dicc_circula,time_da):
    
    vh= ' nny '
    
    if (1500<ll[0]<2400 or  3400<ll[0]<3700)and not good_fit:
        #print '--;--'
        return vh
    if (5400<ll[0]<5800):
        #print '--;--'
        return vh
    
    
    evento_no_reportado=(ll[0] not in report_dicc_circula) or not ((report_dicc_circula[ll[0]])['reportado'])
    evento_reportado=not(evento_no_reportado)
    
    if good_fit and h<=500:
        
        if 8<np.abs(velocidad)<12 and (st_cant>16) and (st=='update'): #No reporta fin
            vh=' retro '
            st_rep,id_rep=handle_report_dicc_from_cluster(ll,report_dicc_circula,st,time_da) #si es la primera vez cambia update por inicio
            q_db.append((st_rep,st_cant,'retro',bin2prog(ll[0]),id_rep))

        if np.abs(velocidad)<=8 and (st_cant>13) and (st=='update'):#No reporta fin
            vh=' retro '
            st_rep,id_rep=handle_report_dicc_from_cluster(ll,report_dicc_circula,st,time_da)          
            q_db.append((st_rep,st_cant,'retro',bin2prog(ll[0]),id_rep))
            
        if 150>np.abs(velocidad)>18.95:
            vh='camioneta '
            if not( (st=='fin') and evento_no_reportado):
                #if st_cant==-1:#si termina el evento y esta listo para reportar
                    #st_rep,id_rep=handle_report_dicc_from_cluster(ll,report_dicc_circula,st,time_da))
                    #q_db.append((st_rep,st_cant,'camioneta',bin2prog(ll[0]),id_rep))
                if st in ['inicio', 'update'] or (st_cant==-1):
                    st_rep,id_rep=handle_report_dicc_from_cluster(ll,report_dicc_circula,st,time_da)
                    q_db.append((st_rep,st_cant,'camioneta',bin2prog(ll[0]),id_rep))
                                    
            else: #caso no soportado por handle_report_dicc
                if st_cant==-1:
                    id_rep=time.time()
                    q_db.append((st,st_cant,'camioneta',bin2prog(ll[0]),id_rep))

        
        if 12<=np.abs(velocidad)<=18.95:
            
            if st_cant>=17 and 13<=np.abs(velocidad)<=16 and (st=='update'):#No reporta fin
                vh=' camion '
                st_rep,id_rep=handle_report_dicc_from_cluster(ll,report_dicc_circula,st,time_da)
                q_db.append((st_rep,st_cant,'camion',bin2prog(ll[0]),id_rep))                

            if st_cant>=19 and 16<np.abs(velocidad)<19 and (st=='update'):#No reporta fin
                vh=' camion '
                st_rep,id_rep=handle_report_dicc_from_cluster(ll,report_dicc_circula,st,time_da)
                q_db.append((st_rep,st_cant,'camion',bin2prog(ll[0]),id_rep))   
            
            if st_cant>=17 and np.abs(velocidad)<=12.99 and (st =='update'):#No reporta fin
                vh=' retro '
                st_rep,id_rep=handle_report_dicc_from_cluster(ll,report_dicc_circula,st,time_da)
                q_db.append((st_rep,st_cant,'retro',bin2prog(ll[0]),id_rep))            
            
            if h<100 and(np.abs(velocidad)>15) and (st=='fin') and (st_cant==-1):#evento sin inicio
                vh=' camioneta '
                id_rep=time.time()
#                handle_report_dicc_from_cluster(ll,report_dicc_circula,st)
                q_db.append((st,st_cant,'camioneta',bin2prog(ll[0]),id_rep))            

        #TODO: CASO FIN -1 para updates <500            
        
    if good_fit and h>500:#casos borde, continuar reportando-No reporta fin porque cambia esquina inferior, posibilidad de cambiar las filas o identificar eventos diferentepara estos casos

        if np.abs(velocidad)<=12.99:
            vh=' retro '
#            st_rep,id_rep=handle_report_dicc_from_cluster(ll,report_dicc_circula,st,time_da)
            q_db.append((st,st_cant,'retro',bin2prog(ll[0]),None))

        if 13<=np.abs(velocidad)<=22:
            vh=' camion '
#            st_rep,id_rep=handle_report_dicc_from_cluster(ll,report_dicc_circula,st,time_da))
            q_db.append((st,st_cant,'camion',bin2prog(ll[0]),None))    

    if not(good_fit) and 80<w<150 and h>1.66*w:#casos borde, continuar reportando No reporta fin
        vh=' retro '
        q_db.append((st,st_cant,'retro',bin2prog(ll[0]),None))
    
    if not(good_fit) and h<=500 and w>20 and st=='fin' and st_cant==-1 and evento_reportado:#lo reporto pero se rompio en el camino, lo rescato
        st_rep,id_rep=handle_report_dicc_from_cluster(ll,report_dicc_circula,st,time_da)
        q_db.append((st_rep,st_cant,'fin evento default',bin2prog(ll[0]),id_rep))
        # inser
    return vh        
                
def insert_report(prog,evt,ts,st_n=1,id_r=None, db_ins=False):

    if db_ins:
        sv="SWPLPGLPAPL07"
        db="Y-NTEGRO"
        uid="yntegro"
        pwd="Fibra2019"
        driver="ODBC Driver 17 for SQL Server" 
        tuple_cnx=(driver,sv,db,uid,pwd)
        cnxn_str='DRIVER={%s};SERVER=%s;DATABASE=%s;uid=%s;pwd=%s' % tuple_cnx
        cnxn = pyodbc.connect(cnxn_str)
        
        
        if id_r==None:
            sample_event=int(time.time())
        else:
            sample_event=id_r
        sample_prog=prog
        sample_st=st_n
        if evt=='camioneta':
            sample_crit=1
        else:
            sample_crit=2
        sample_date=ts
        sample_zone=0
        sample_prox=0.0
        sample_intmax=0
        sample_tag=evt
        sample_conf=1.0
        sample_id=0
        sample_path='TBD'
        
        sample_tuple=(sample_event,sample_prog,sample_st,sample_crit,sample_date,sample_zone,sample_prox,sample_intmax,sample_tag,sample_conf,sample_id,sample_path)
        insert_str='INSERT INTO dbo.Eventos (id_evento, progresiva, id_estado, criticidad, fecha_evento, zona, prox_ducto, inten_max, descripcion, confiabilidad,id_estado_sistema,path_muestra) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)'
        with cnxn.cursor() as crs:
            crs.execute(insert_str,sample_tuple)
        cnxn.close() #context manager maybe

    
def db_rep_th(q):
    
    
    excav_report={}
    
    camioneta_report={}
    
    retro_cam_report={}
    
    for i in range(330,360):
        excav_report[i]=datetime.datetime(1968,2,3,4,5,6)
        camioneta_report[i]=datetime.datetime(1968,2,3,4,5,6)
        retro_cam_report[i]=datetime.datetime(1968,2,3,4,5,6)
    
    evt_diccs={'retro':retro_cam_report,'camion':retro_cam_report,'camioneta':camioneta_report,'excavacion':excav_report}

    print ("Comienza reporteDB")    
    while True:
        
        if len(q)>0:
            
            st, st_cant, rep_type, prog,id_r=q.popleft()
            print (st, st_cant,rep_type,prog,id_r)
            
            rn=datetime.datetime.now()
            time_wait_evt_seg=0
            
            evt=rep_type.replace(' ','')
            
            if evt!='fineventodefault':
                
                dicc=evt_diccs[evt]
            
                last_rep_prog=(rn-dicc[round(prog)]).seconds
                last_rep_prog_sig=(rn-dicc[round(prog+1)]).seconds
                last_rep_prog_prev=(rn-dicc[round(prog-1)]).seconds
            
                print ('evento', evt,rn )
                
            if evt not in ['excavacion','fineventodefault']:
                report_prog=(last_rep_prog>time_wait_evt_seg)and(last_rep_prog_sig>time_wait_evt_seg)and(last_rep_prog_prev>time_wait_evt_seg)
            else:
                report_prog=(rn-dicc[round(prog)]).seconds>0
            
            
            if report_prog and (st not in ['fin','discard']):
                try:
                    if st=='inicio' or st=='rep_exc':
                        insert_report(prog,evt,rn,st_n=1,id_r=id_r)
                    if st=='update':
                        insert_report(prog,evt,rn,st_n=2,id_r=id_r)
                    print ("reportado", evt, rn)
                except:
                    print ("falla db")
                dicc[round(prog)]=rn
            
            if st=='fin' and st_cant==-1:#FIN SOLO SE REPORTA el -1 (cuando pasa la linea de y=50)
                try:
                    insert_report(prog,evt,rn,st_n=3,id_r=id_r)
                    print ("fin report",evt,rn)
                except:
                    print ("falla db")
                
           
            
            
        else:
            time.sleep(0.1)
            
            #if st=='fin':
                
                
            
            
    
