"""Synthetic contract checks; no match ground truth is loaded."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from . import config as c
from .layer5_decoder import build
from .layer6_output import write

class PipelineInvariants(unittest.TestCase):
    def fixture(self, n=300):
        e={name:np.full(n,-1,dtype=int) for name in ('owner','reception','kick')}
        e.update({name:np.zeros(n,dtype=bool) for name in ('duel','glitch','gap_edge')})
        e['owner'][:150]=0; e['owner'][150:]=1
        return e,np.zeros(n,dtype=bool),np.full(n,-1,dtype=int)

    def test_cannot_switch_without_event(self):
        e,dead,restart=self.fixture()
        with patch.object(c,'N',300),patch.object(c,'HALVES',((0,300),)):
            labels=build({},pd.DataFrame(),e,dead,restart)
        self.assertEqual(len(np.flatnonzero(labels[1:]!=labels[:-1])),0)

    def test_reception_allows_switch_but_duel_does_not(self):
        e,dead,restart=self.fixture(); e['reception'][150]=1
        with patch.object(c,'N',300),patch.object(c,'HALVES',((0,300),)):
            labels=build({},pd.DataFrame(),e,dead,restart)
            self.assertTrue((labels[:150]=='H').all())
            self.assertTrue((labels[150:]=='A').all())
            e['duel'][150]=True
            labels=build({},pd.DataFrame(),e,dead,restart)
            self.assertTrue((labels==labels[0]).all())

    def test_dead_frames_are_not_possession(self):
        e,dead,restart=self.fixture();dead[100:200]=True
        with patch.object(c,'N',300),patch.object(c,'HALVES',((0,300),)):
            labels=build({},pd.DataFrame(),e,dead,restart)
        self.assertTrue((labels[100:200]=='D').all())

    def test_serialization_masks_and_repeat_hash(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT) as folder:
            path=Path(folder)/'predictions.csv'
            with patch.object(c,'N',10),patch.object(c,'HALVES',((2,5),(7,9))):
                first=write(np.full(10,'H'),path)
                second=write(np.full(10,'H'),path)
            self.assertEqual(first,second)
            lines=path.read_text().splitlines()
            self.assertEqual(len(lines),11)
            self.assertEqual(lines[0],'frame,label')
            self.assertEqual(lines[1],'0,X')
            self.assertEqual(lines[3],'2,H')

if __name__=='__main__': unittest.main()
