from engine.promql_analyzer import PromQLAnalyzer

class AnalyzerManager:
    
    sync_interval: str = None
    retrain_interval: str = None
    
    
    def __init__(self):
        self.analyzers = []

    def add_analyzer(self, analyzer):
        """添加一个 PromQLAnalyzer 实例到管理器中"""
        if isinstance(analyzer, PromQLAnalyzer):
            self.analyzers.append(analyzer)
        else:
            raise TypeError("Expected a PromQLAnalyzer instance")

    def remove_analyzer(self, analyzer):
        """从管理器中移除一个 PromQLAnalyzer 实例"""
        if analyzer in self.analyzers:
            self.analyzers.remove(analyzer)

    def sync_all(self):
        """同步所有的 PromQLAnalyzer 实例"""
        for analyzer in self.analyzers:
            analyzer.sync()


    def list_analyzers(self):
        """列出所有的 PromQLAnalyzer 实例"""
        return self.analyzers

# 使用示例
manager = AnalyzerManager()
analyzer1 = PromQLAnalyzer("up", 60, "http://localhost:9090")
analyzer2 = PromQLAnalyzer("node_memory_MemAvailable_bytes", 120, "http://localhost:9090")

manager.add_analyzer(analyzer1)
manager.add_analyzer(analyzer2)

# 同步所有分析器
manager.sync_all()

