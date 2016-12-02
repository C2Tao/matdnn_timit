import os
import numpy as np

root = '/home/c2tao/TIMIT/timit/'

def get_timit_list():
    timit_list = []
    for type in sorted(['train','test']):
        for dir0 in sorted(os.listdir(root+type+'/')):
            for dir1 in sorted(os.listdir(root+type+'/'+dir0+'/')):
                for dir2 in sorted(os.listdir(root+type+'/'+dir0+'/'+dir1+'/')):
                    if 'phn' in dir2: 
                        name = type+'_'+dir0+'_'+dir1+'_'+dir2[:-4]
                        timit_list.append(name)
    return sorted(timit_list)

def timit_path(name):
    return os.path.join(root,*name.split('_'))

def timit_dur(name):
    '''
    try:
        return float(open(timit_path(name)+'.txt','r').readlines()[0].split()[1])
    except:
        print "something missing with in ", name, ".txt"
        return float(open(timit_path(name)+'.phn','r').readlines()[-1].split()[1])
        # test_dr3_fpkt0_sil_1538.wrd
    '''
    if name=='test_dr3_fpkt0_si1538': 
        return float(open(timit_path(name)+'.phn','r').readlines()[-1].split()[1])
    if name=='train_dr4_fpaf0_sx154':
        return float(open(timit_path(name)+'.phn','r').readlines()[-1].split()[1])
    return float(open(timit_path(name)+'.phn','r').readlines()[-1].split()[1])
    #return float(open(timit_path(name)+'.txt','r').readlines()[0].split()[1])


def timit_parse(name, dot):
    #dot = word or phn
    tuples = []
    for line in open(timit_path(name)+'.'+dot,'r'):
        beg, end, tok = line.strip().split()
        tuples.append([float(beg), float(end), tok])
    beg_list, end_list, tag_list = zip(*tuples)
    tag_list = list(tag_list)
    int_list = list(end_list)
    dur = timit_dur(name)
    
    if beg_list[0]>0:
        int_list.insert(0, beg_list[0])
        tag_list.insert(0, 'sil')
    if end_list[-1]<dur:
        int_list.append(dur) 
        tag_list.append('sil') 
    while True:
        for i in range(1, len(int_list)):
            if int_list[i-1]>=int_list[i]:
                print "warning: the alignment of", name, dot," is abnormal"
                print tag_list[i-1], int_list[i-1]
                print tag_list[i], int_list[i]
                print int_list
                print tag_list
                int_list.pop(i-1)
                tag_list.pop(i-1)
                print int_list
                print tag_list
            break
        break
        
    int_list = np.array(int_list)/dur
    return int_list, tag_list 


def timit_mlf(name, ostream, dot, dur):
    ostream.write('"*/'+name+'.rec"\n')
    int_list, tag_list = timit_parse(name, dot)
    full = np.insert(int_list, 0, 0.0)
    for i in range(len(tag_list)):
        ostream.write('{} {} {} {}\n'.format(int(full[i]*dur)*100000, int(full[i+1]*dur)*100000, tag_list[i], 0.0))
    ostream.write('.\n')    

def timit_write(out_file, dot, opt):
    timit_list = get_timit_list()
    from zrst.util import MLF
    mlf = MLF('timit_dummy_train.mlf')
    mlf.merge(MLF('timit_dummy_test.mlf'))
    print mlf.wav_list
    dur_dict = dict(zip(mlf.wav_list, mlf.int_list))
    with open(out_file, 'w') as f:
        f.write('#!MLF!#\n')
        for t in timit_list:
            print t    
            if opt in t:
                timit_mlf(t, f, dot, dur_dict[t][-1])
    
            

#timit_list = get_timit_list()
#print timit_parse(timit_list[0], 'phn')
timit_write('timit_train_wrd.mlf', 'wrd', 'train')
timit_write('timit_test_wrd.mlf', 'wrd', 'test')
timit_write('timit_train_phn.mlf', 'phn', 'train')
timit_write('timit_test_phn.mlf', 'phn', 'test')
