import unittest
import torch
from whole_body_jepa.native_event_encoder import NativeEventFilter


class NativeEventEncoderTest(unittest.TestCase):
    def test_inactive_nonzero_events_do_not_change_state_or_clock(self):
        torch.manual_seed(31)
        model = NativeEventFilter(12, 8)
        target = torch.randn(1, 2, 12)
        target[..., -2] = 0
        target[0, :, -1] = torch.tensor([.1, .9])
        expected = model(target, torch.ones(1, 2, dtype=torch.bool))
        other = torch.randn(1, 5, 12) * 100
        other[..., -2] = 0
        other[0, :, -1] = torch.tensor([-.5, .2, .4, .7, 1.5])
        sequence = torch.cat([other[:, :1], target[:, :1], other[:, 1:4], target[:, 1:], other[:, 4:]], dim=1)
        mask = torch.tensor([[False, True, False, False, False, True, False]])
        actual = model(sequence, mask)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_rate_event_updates_but_padding_does_not(self):
        torch.manual_seed(7);model=NativeEventFilter(12,8)
        event=torch.zeros(1,1,12);event[0,0,-2]=1.;event[0,0,-1]=.2
        mask=torch.zeros(1,1,dtype=torch.bool)
        h=model(event,mask)[0]
        self.assertGreater(float(h.detach().abs().sum()),0.)
        padded=torch.cat([torch.zeros(1,2,12),event,torch.zeros(1,2,12)],dim=1)
        hp=model(padded,torch.zeros(1,5,dtype=torch.bool))[0]
        torch.testing.assert_close(h,hp,atol=0,rtol=0)
        empty=model(torch.zeros_like(event),mask)[0]
        torch.testing.assert_close(empty,torch.zeros_like(empty),atol=0,rtol=0)
        with self.assertRaises(ValueError):model(event)


if __name__=='__main__':unittest.main()
