"""Makale değerlendirmesi: dört HisNet sistemi ve Jev.

Varsayılan çalıştırma çevrimdışı regresyonları yürütür.
Gerçek makale koşusu için: python -m tests.test_paper_evaluation_classical --live
"""
import sys
import unittest
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import test_paper_evaluation as shared


SYSTEMS = ("hisnet", "hisnet_dp", "dp_hisnet", "lw_hisnet_dp", "jev")
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

    def test_paper_groups_keep_jev_with_hisnet(self):
        from tests.test_paper_evaluation_llm import SYSTEMS as OLLAMA_SYSTEMS

        self.assertEqual(SYSTEMS, ("hisnet", "hisnet_dp", "dp_hisnet", "lw_hisnet_dp", "jev"))
        self.assertEqual(OLLAMA_SYSTEMS, ("llm", "dp_llm"))
        self.assertCountEqual(SYSTEMS + OLLAMA_SYSTEMS, shared.SYSTEMS)


if __name__ == "__main__":
    shared.run_test_file(run_paper)
