"""项目包最小导入测试。"""


def test_package_import() -> None:
    """确认安装路径和公开版本正常。"""

    import replenishment

    assert replenishment.__version__ == "0.4.0"
