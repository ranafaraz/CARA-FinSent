def test_imports():
    import cara_finsent
    from cara_finsent.feature_extractor import FinancialFeatureExtractor
    assert FinancialFeatureExtractor().extract_one('Revenue rose 20%')['has_numeric_signal'] == 1
