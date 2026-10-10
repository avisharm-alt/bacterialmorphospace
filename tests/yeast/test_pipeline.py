import numpy as np
import pandas as pd
import pytest
from yeast_morphology.data import sequence_features, proteins, read_table
from yeast_morphology.splits import make_split
from yeast_morphology.models import Representation, fit_neural, neural_samples
from yeast_morphology.metrics import per_row_scores, paired_bootstrap


def fixture_data():
    rng=np.random.default_rng(13);n=240
    y=rng.normal(size=(n,6));y[:35,:2]=2
    # Both individual components occur outside the held-out intersection.
    y[35:75,0]=2;y[35:75,1]=0
    y[75:115,0]=0;y[75:115,1]=2
    meta=pd.DataFrame({'id':[f'single:g{i}' for i in range(n)],'gene':[f'g{i}' for i in range(n)],
        'background':'single','sequence_group':[f's{i}' for i in range(n)]})
    return meta,y,['C11-1_A','C115_A','a','b','c','d']


def test_composition_split_is_complete_and_group_safe():
    m,y,t=fixture_data();m.loc[80,'sequence_group']=m.loc[0,'sequence_group']
    split=make_split(m,y,t,'composition')
    assert split.loc[80,'partition']=='buffer'
    assert np.all(y[split.partition=='test',:2]>1)
    assert not (y[split.partition.isin(['train','val']),:2]>1).all(1).any()
    sets=[set(split.loc[split.partition==p,'sequence_group']) for p in ['train','val','test']]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    pd.testing.assert_frame_equal(split,make_split(m,y,t,'composition'))


def test_gene_split_keeps_sequence_duplicates_together():
    m,y,t=fixture_data();m.loc[1,'sequence_group']=m.loc[0,'sequence_group']
    s=make_split(m,y,t,'gene')
    assert s.loc[0,'partition']==s.loc[1,'partition']


def test_background_holds_combinations_but_keeps_single_background():
    m,y,t=fixture_data();q=m.copy();q['id']=q.id.str.replace('single:','triple:');q['background']='triple'
    both=pd.concat([m,q],ignore_index=True);s=make_split(both,np.r_[y,y],t,'background')
    assert (s.loc[s.background=='single','partition']=='train').all()
    assert set(s.loc[s.partition=='test','gene'])<=set(s.loc[s.partition=='train','gene'])
    tr=set(s.loc[(s.background=='triple')&(s.partition=='train'),'sequence_group'])
    te=set(s.loc[s.partition=='test','sequence_group'])
    assert not tr&te


def test_representation_is_fitted_on_train_only():
    rng=np.random.default_rng(0);x=rng.normal(size=(80,6));y=rng.normal(size=(80,10));y[0,0]=np.nan
    r=Representation().fit(x[:60],y[:60],5)
    means=r.yscale.mean_.copy();basis=r.pca.components_.copy()
    r.x(x[60:]*1000);r.y(y[60:]*1000)
    np.testing.assert_equal(means,r.yscale.mean_);np.testing.assert_equal(basis,r.pca.components_)
    assert np.isfinite(r.y(y)).all()


def test_missing_targets_never_imputed_into_scores():
    y=np.array([[1.,np.nan]]);samples=np.array([[[1.,-9999.],[1.,9999.]]])
    s=per_row_scores(y,samples)
    for k in ['mse','crps','energy','width90']:assert s[k][0]==0
    assert s['coverage90'][0]==1


def test_crps_and_energy_known_distribution():
    # Forecast {0,2}, observation 1: E|X-y| - .5 E|X-X'| = 1 - .5 = .5.
    s=per_row_scores(np.array([[1.]]),np.array([[[0.],[2.]]]))
    assert s['crps'][0]==pytest.approx(.5)
    assert s['energy'][0]==pytest.approx(.5)
    assert s['mse'][0]==0


def test_cluster_bootstrap_identical_predictors():
    r=paired_bootstrap(np.arange(4.),np.arange(4.),['a','a','b','b'],n_boot=50)
    assert r['difference']==0 and r['ci95']==[0,0]


def test_sequence_features_are_fixed_and_normalized():
    x=sequence_features('ACDACD*X');assert x.shape==(421,)
    assert x[1:21].sum()==pytest.approx(1)
    assert x[21:].sum()==pytest.approx(1)
    np.testing.assert_equal(x,sequence_features('ACDACD*X'))


def test_table_missing_sentinels_and_fasta(tmp_path):
    p=tmp_path/'table.tsv';p.write_text('ORF\tx\ty\nYAL001C\t-1\t2\n')
    assert np.isnan(read_table(p).iloc[0,0])
    p=tmp_path/'proteins.fasta';p.write_text('>YAL001C label\nACD*\n>YAL002W\nAAA\n')
    assert proteins(p)=={'YAL001C':'ACD','YAL002W':'AAA'}


@pytest.mark.parametrize('kind',['gaussian','diffusion'])
def test_neural_training_sampling_roundtrip(kind):
    import torch
    torch.set_num_threads(2)
    rng=np.random.default_rng(42);x=rng.normal(size=(70,5)).astype('float32');z=x[:,:3]*.5
    ck=fit_neural(kind,x[:50],z[:50],x[50:],z[50:],epochs=2,steps=10,width=32,max_seconds=15)
    a=neural_samples(ck,x[50:55],4,123);b=neural_samples(ck,x[50:55],4,123)
    assert a.shape==(5,4,3) and np.isfinite(a).all()
    np.testing.assert_equal(a,b)


def test_embedding_loader_preserves_row_alignment(tmp_path):
    from yeast_morphology.data import load
    prepared=tmp_path/'prepared';prepared.mkdir()
    pd.DataFrame({'id':['single:B','triple:A','single:B'],'gene':['B','A','B']}).to_csv(prepared/'metadata.tsv',sep='\t',index=False)
    np.savez_compressed(prepared/'dataset.npz',ids=['single:B','triple:A','single:B'],
                        x=[[0,0],[0,1],[0,0]],y=[[1],[2],[3]],traits=['area'])
    emb=tmp_path/'emb.npz';np.savez_compressed(emb,genes=['A','B'],embeddings=[[10,20],[30,40]])
    x,y,traits,meta=load(tmp_path,emb)
    np.testing.assert_array_equal(x,[[30,40,0],[10,20,1],[30,40,0]])
    np.testing.assert_array_equal(y,[[1],[2],[3]])


def test_inference_rejects_untrained_background_before_prediction(tmp_path):
    import pickle
    from yeast_morphology.experiment import predict
    with (tmp_path/'bundle.pkl').open('wb') as f:
        pickle.dump({'train_backgrounds':['single']},f)
    with pytest.raises(ValueError,match='absent from model training'):
        predict(tmp_path,tmp_path,['YAL002W'],background='triple',out=tmp_path/'out')
