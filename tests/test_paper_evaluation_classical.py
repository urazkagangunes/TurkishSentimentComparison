"""Makale değerlendirmesi: dört klasik HisNet sistemi.

Varsayılan çalıştırma çevrimdışı regresyonları yürütür.
Gerçek makale koşusu için: python -m tests.test_paper_evaluation_classical --live
"""
import sys
import unittest
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import test_paper_evaluation as shared


SYSTEMS = ("hisnet", "hisnet_dp", "dp_hisnet", "lw_hisnet_dp")
DatasetTests = shared.DatasetTests
MetricTests = shared.MetricTests


def run_paper(argv):
    return shared.run_paper_group(argv, SYSTEMS, "paper_classical")


class RunnerTests(shared.PaperRunnerChecks, shared.ClassicalRunnerChecks, unittest.TestCase):
    systems = SYSTEMS
    pair_count = 4


class SuiteTests(shared.PaperSuiteChecks, unittest.TestCase):
    systems = SYSTEMS
    run_cli = staticmethod(run_paper)


if __name__ == "__main__":
    shared.run_test_file(run_paper)
