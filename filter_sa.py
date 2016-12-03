from zrst.util import MLF

def filter_sa(input_mlf_name, output_mlf_name):
    mlf = MLF(input_mlf_name)
    nosa_indices = []
    nosa_wavname = []
    for i,w in enumerate(mlf.wav_list):
        if "_sa" not in w:
            nosa_indices.append(i)
            nosa_wavname.append(w)

    mlf.write(output_mlf_name, selection=zip(nosa_indices, nosa_wavname))

filter_sa('timit_train_phn.mlf', 'timit_train_phn_nosa.mlf')
filter_sa('timit_train_wrd.mlf', 'timit_train_wrd_nosa.mlf')
filter_sa('timit_test_phn.mlf', 'timit_test_phn_nosa.mlf')
filter_sa('timit_test_wrd.mlf', 'timit_test_wrd_nosa.mlf')
