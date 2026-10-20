import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('prepare_wrench6',Path(__file__).parents[1]/'scripts/prepare_wrench6_training.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_split_40_10_5_keeps_records_unique_and_sessions_represented():
    ids=[f'Tianji_Marvin_M6CCS_261020_{hour}{i:04d}_0' for hour,n in [('17',25),('19',30)] for i in range(n)]
    groups={key:'sha_'+key for key in ids}
    split=module.make_split(ids,groups)
    roles=['train','validation','test']
    assert [len(split[r+'_episode_ids']) for r in roles]==[40,10,5]
    assert len(set(sum([split[r+'_episode_ids'] for r in roles],[])))==55
    assert split==module.make_split(list(reversed(ids)),groups)
    assert all({key.rsplit('_',2)[1][:2] for key in split[r+'_episode_ids']}=={'17','19'} for r in roles)
    with pytest.raises(ValueError):module.make_split(ids,{key:'same' for key in ids})


def test_distribution_keeps_peak_and_channel_scales():
    values=np.zeros((100,6));values[-1]=[100,200,300,1,2,3]
    report=module.distribution(values)
    assert report['quantiles']['1']==[100,200,300,1,2,3]
    assert report['median']==[0]*6 and report['std'][0]>report['std'][3]
