"""修补覆盖核对端到端测试（FastAPI TestClient + 临时 SQLite 文件）。"""

import os
import tempfile
import unittest


class RepairCoverageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        db_file = os.path.join(self._tmp.name, "test.db")
        os.environ["CRACK_DB_PATH"] = db_file

        from fastapi.testclient import TestClient
        from app import db as db_mod
        from app.main import app

        db_mod.DB_PATH = db_file
        db_mod.init_db()
        self.client = TestClient(app)

    def tearDown(self):
        self._tmp.cleanup()

    def _obs(self, oid, r0, r1, c0, c1, grade=2, conf=0.8, inspection="insp-1"):
        return {
            "observation_id": oid,
            "inspection_id": inspection,
            "radius_start": r0,
            "radius_end": r1,
            "clock_start": c0,
            "clock_end": c1,
            "grade": grade,
            "confidence": conf,
        }

    def _region(self, r0, r1, c0, c1):
        return {
            "radius_start": r0,
            "radius_end": r1,
            "clock_start": c0,
            "clock_end": c1,
        }

    def _write(self, batch_id, observations, turbine="WT-01", blade="B3"):
        resp = self.client.post("/api/v1/observations/batch", json={
            "batch_id": batch_id,
            "turbine_id": turbine,
            "blade_id": blade,
            "observations": observations,
        })
        self.assertEqual(resp.status_code, 200, resp.text)

    def _coverage(self, regions, inspection="insp-1", turbine="WT-01", blade="B3"):
        return self.client.post("/api/v1/repair-coverage", json={
            "turbine_id": turbine,
            "blade_id": blade,
            "inspection_id": inspection,
            "repair_regions": regions,
        })

    def _item(self, body, oid):
        return next(o for o in body["observations"] if o["observation_id"] == oid)

    # --------------------------------------------------------- 覆盖计算

    def test_partial_cover_and_overlap_counted_once(self):
        # 观察 [0,10] × [1,3]：面积 10m × 2h = 20
        self._write("b1", [self._obs("o1", 0, 10, 1, 3)])
        # 两个修补区域在 [4,6]×[2,3] 处重叠，重叠部分只计一次：
        # 并集覆盖 = [2,4]×2h + [4,6]×2h + [6,8]×1h = 4+4+2 = 10
        resp = self._coverage([
            self._region(2, 6, 1, 3),
            self._region(4, 8, 2, 4),
        ])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["repair_region_count"], 2)
        self.assertEqual(body["observation_count"], 1)
        self.assertEqual(body["checked_count"], 1)
        self.assertEqual(body["zero_area_count"], 0)

        item = self._item(body, "o1")
        self.assertAlmostEqual(item["area"], 20.0)
        self.assertAlmostEqual(item["covered_area"], 10.0)
        self.assertAlmostEqual(item["coverage_ratio"], 0.5)
        self.assertEqual(item["status"], "partial")
        self.assertAlmostEqual(body["total_area"], 20.0)
        self.assertAlmostEqual(body["total_covered_area"], 10.0)
        self.assertAlmostEqual(body["overall_coverage_ratio"], 0.5)

    def test_full_cover(self):
        self._write("b2", [self._obs("o1", 0, 10, 1, 3)])
        resp = self._coverage([self._region(0, 10, 1, 3)])
        self.assertEqual(resp.status_code, 200, resp.text)
        item = self._item(resp.json(), "o1")
        self.assertAlmostEqual(item["covered_area"], 20.0)
        self.assertAlmostEqual(item["coverage_ratio"], 1.0)
        self.assertEqual(item["status"], "full")

    def test_uncovered(self):
        self._write("b3", [self._obs("o1", 0, 10, 1, 3)])
        resp = self._coverage([self._region(20, 30, 5, 6)])
        self.assertEqual(resp.status_code, 200, resp.text)
        item = self._item(resp.json(), "o1")
        self.assertAlmostEqual(item["covered_area"], 0.0)
        self.assertAlmostEqual(item["coverage_ratio"], 0.0)
        self.assertEqual(item["status"], "uncovered")

    def test_region_wrapping_twelve(self):
        # 观察跨 12 点：[5,15] × 11->1，面积 10m × 2h = 20
        self._write("b4", [self._obs("o1", 5, 15, 11, 1)])
        # 修补区域同样跨 12 点：11.5->0.5，覆盖其中 1 小时
        resp = self._coverage([self._region(5, 15, 11.5, 0.5)])
        self.assertEqual(resp.status_code, 200, resp.text)
        item = self._item(resp.json(), "o1")
        self.assertAlmostEqual(item["area"], 20.0)
        self.assertAlmostEqual(item["covered_area"], 10.0)
        self.assertEqual(item["status"], "partial")

    def test_clock_twelve_normalized_like_observation(self):
        # 观察写入时 12 归一为 0；修补区域 12 点写法应得到一致结果
        self._write("b5", [self._obs("o1", 0, 4, 11, 12)])
        resp = self._coverage([self._region(0, 4, 11, 12)])
        self.assertEqual(resp.status_code, 200, resp.text)
        item = self._item(resp.json(), "o1")
        self.assertAlmostEqual(item["area"], 4.0)
        self.assertAlmostEqual(item["covered_area"], 4.0)
        self.assertEqual(item["status"], "full")

    def test_multiple_observations_mixed_status(self):
        self._write("b6", [
            self._obs("full", 0, 4, 1, 2),      # 4
            self._obs("part", 10, 14, 1, 2),    # 4，只盖一半
            self._obs("none", 20, 24, 5, 6),    # 4，不相交
        ])
        resp = self._coverage([
            self._region(0, 4, 1, 2),
            self._region(12, 14, 1, 2),
        ])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(self._item(body, "full")["status"], "full")
        part = self._item(body, "part")
        self.assertEqual(part["status"], "partial")
        self.assertAlmostEqual(part["covered_area"], 2.0)
        self.assertEqual(self._item(body, "none")["status"], "uncovered")

    # --------------------------------------------------------- 零面积

    def test_zero_area_observations_listed_separately(self):
        self._write("b7", [
            self._obs("ok", 0, 10, 1, 3),
            self._obs("zero-r", 3, 3, 1, 2),   # 径向零长
            self._obs("zero-c", 3, 5, 4, 4),   # 钟位单点
        ])
        resp = self._coverage([self._region(0, 10, 1, 3)])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["observation_count"], 3)
        self.assertEqual(body["checked_count"], 1)
        self.assertEqual(body["zero_area_count"], 2)
        zero_ids = {z["observation_id"] for z in body["zero_area_observations"]}
        self.assertEqual(zero_ids, {"zero-r", "zero-c"})
        for z in body["zero_area_observations"]:
            self.assertEqual(z["area"], 0.0)
        # 零面积观察不出现在核对列表中
        self.assertEqual({o["observation_id"] for o in body["observations"]},
                         {"ok"})

    # --------------------------------------------------------- 边界语义

    def test_empty_regions_means_no_repair(self):
        self._write("b8", [self._obs("o1", 0, 10, 1, 3)])
        resp = self._coverage([])
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["repair_region_count"], 0)
        item = self._item(body, "o1")
        self.assertEqual(item["status"], "uncovered")
        self.assertAlmostEqual(item["covered_area"], 0.0)
        self.assertAlmostEqual(body["overall_coverage_ratio"], 0.0)

    def test_filters_by_inspection(self):
        self._write("b9a", [self._obs("a1", 0, 10, 1, 3, inspection="insp-1")])
        self._write("b9b", [self._obs("a2", 0, 10, 1, 3, inspection="insp-2")])
        resp = self._coverage([self._region(0, 10, 1, 3)], inspection="insp-1")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["observation_count"], 1)
        self.assertEqual({o["observation_id"] for o in body["observations"]},
                         {"a1"})

    def test_inspection_not_found_404(self):
        self._write("b10", [self._obs("o1", 0, 10, 1, 3)])
        resp = self._coverage([self._region(0, 10, 1, 3)], inspection="nope")
        self.assertEqual(resp.status_code, 404)
        # 叶片本身也不存在时同样 404
        resp = self._coverage([], turbine="WT-X", blade="B9")
        self.assertEqual(resp.status_code, 404)

    def test_invalid_region_422(self):
        self._write("b11", [self._obs("o1", 0, 10, 1, 3)])
        # radius_end < radius_start
        self.assertEqual(self._coverage([
            {"radius_start": 5, "radius_end": 2,
             "clock_start": 1, "clock_end": 2},
        ]).status_code, 422)
        # 钟位越界
        self.assertEqual(self._coverage([
            self._region(0, 5, 1, 13),
        ]).status_code, 422)
        # 负径向
        self.assertEqual(self._coverage([
            self._region(-1, 5, 1, 2),
        ]).status_code, 422)
        # 多余字段
        self.assertEqual(self._coverage([
            dict(self._region(0, 5, 1, 2), region_id="r1"),
        ]).status_code, 422)
        # 缺字段
        self.assertEqual(self._coverage([
            {"radius_start": 0, "radius_end": 5, "clock_start": 1},
        ]).status_code, 422)

    # --------------------------------------------------------- 只读语义

    def test_read_only_does_not_modify_state(self):
        self._write("b12", [
            self._obs("o1", 0, 10, 1, 3),
            self._obs("o2", 8, 18, 2, 4, grade=3),
        ])
        review_before = self.client.get("/api/v1/reviews/WT-01/B3").json()
        adj_before = self.client.get("/api/v1/adjudications/WT-01/B3").json()

        resp = self._coverage([self._region(0, 10, 1, 3)])
        self.assertEqual(resp.status_code, 200, resp.text)

        review_after = self.client.get("/api/v1/reviews/WT-01/B3").json()
        adj_after = self.client.get("/api/v1/adjudications/WT-01/B3").json()
        self.assertEqual(review_before, review_after)
        self.assertEqual(adj_before, adj_after)

        # 重复调用结果一致（幂等的只读计算）
        resp2 = self._coverage([self._region(0, 10, 1, 3)])
        self.assertEqual(resp.json(), resp2.json())


if __name__ == "__main__":
    unittest.main()
