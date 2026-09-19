import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProjectContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "train_model.py").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_required_architecture_components_exist(self):
        classes = {
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ClassDef)
        }

        expected = {
            "MultiScaleCNN",
            "ResidualAttentionFusion",
            "TransformerContext",
            "AdaptiveFusion",
            "HybridXRayNet",
        }

        self.assertTrue(expected.issubset(classes))

    def test_reproducibility_function_exists(self):
        functions = {
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("seed_everything", functions)

    def test_cli_defaults_are_present(self):
        self.assertIn("--batch-size", self.source)
        self.assertIn("default=16", self.source)
        self.assertIn("--image-size", self.source)
        self.assertIn("default=224", self.source)
        self.assertIn("--seed", self.source)
        self.assertIn("default=42", self.source)

    def test_architecture_asset_exists(self):
        self.assertTrue(
            (ROOT / "assets" / "hybridxraynet_architecture1.png").exists()
        )

    def test_core_dependency_pins_exist(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        for line in [
            "torch==2.3.1",
            "torchvision==0.18.1",
            "numpy==1.26.4",
            "Flask==3.0.3",
        ]:
            self.assertIn(line, requirements)


if __name__ == "__main__":
    unittest.main()
