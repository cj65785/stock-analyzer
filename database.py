# database.py
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

class Database:
    def __init__(self, db_path='results.db'):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name TEXT NOT NULL,
                dart_report TEXT,
                dart_result TEXT,
                dart_error TEXT,
                news_count INTEGER,
                news_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT '완료'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_result(self, company_name: str, dart_report: str, dart_result: str, 
                   dart_error: str, news_count: int, news_result: str):
        """분석 결과 추가"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results 
            (company_name, dart_report, dart_result, dart_error, news_count, news_result)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (company_name, dart_report, dart_result, dart_error, news_count, news_result))
        
        conn.commit()
        conn.close()
    
    def get_all_results(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """전체 결과 조회 (최신순)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM analysis_results 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def search_results(self, keyword: str) -> List[Dict]:
        """종목명 검색"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM analysis_results 
            WHERE company_name LIKE ?
            ORDER BY created_at DESC
        ''', (f'%{keyword}%',))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def delete_result(self, result_id: int):
        """결과 삭제"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM analysis_results WHERE id = ?', (result_id,))
        
        conn.commit()
        conn.close()
    
    def get_count(self) -> int:
        """전체 결과 개수"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM analysis_results')
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame 변환 (엑셀 다운로드용)"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM analysis_results ORDER BY created_at DESC', conn)
        conn.close()
        return df