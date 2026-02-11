# database.py
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import os

class Database:
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string or os.environ.get('DATABASE_URL')
        if not self.connection_string:
            raise ValueError("DATABASE_URL이 설정되지 않았습니다.")
        
        self.init_db()
    
    def get_connection(self):
        """데이터베이스 연결"""
        return psycopg2.connect(self.connection_string)
    
    def init_db(self):
        """데이터베이스 초기화"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id SERIAL PRIMARY KEY,
                company_name TEXT NOT NULL,
                dart_report TEXT,
                dart_result TEXT,
                dart_error TEXT,
                news_count INTEGER,
                news_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT '완료',
                is_bookmarked BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # is_bookmarked 컬럼이 없으면 추가 (기존 테이블 대응)
        try:
            cursor.execute('''
                ALTER TABLE analysis_results 
                ADD COLUMN IF NOT EXISTS is_bookmarked BOOLEAN DEFAULT FALSE
            ''')
        except:
            pass
        
        # 인덱스 생성
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_company_name 
            ON analysis_results(company_name)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at 
            ON analysis_results(created_at DESC)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_bookmarked 
            ON analysis_results(is_bookmarked)
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def add_result(self, company_name: str, dart_report: str, dart_result: str, 
                   dart_error: str, news_count: int, news_result: str):
        """분석 결과 추가"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results 
            (company_name, dart_report, dart_result, dart_error, news_count, news_result)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (company_name, dart_report, dart_result, dart_error, news_count, news_result))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def get_all_results(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """전체 결과 조회 (최신순)"""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('''
            SELECT * FROM analysis_results 
            ORDER BY created_at DESC 
            LIMIT %s OFFSET %s
        ''', (limit, offset))
        
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return [dict(row) for row in results]
    
    def get_bookmarked_results(self) -> List[Dict]:
        """북마크된 결과만 조회"""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('''
            SELECT * FROM analysis_results 
            WHERE is_bookmarked = TRUE
            ORDER BY created_at DESC
        ''')
        
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return [dict(row) for row in results]
    
    def toggle_bookmark(self, result_id: int):
        """북마크 토글"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE analysis_results 
            SET is_bookmarked = NOT is_bookmarked 
            WHERE id = %s
        ''', (result_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def search_results(self, keyword: str) -> List[Dict]:
        """종목명 검색"""
        conn = self.get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('''
            SELECT * FROM analysis_results 
            WHERE company_name LIKE %s
            ORDER BY created_at DESC
        ''', (f'%{keyword}%',))
        
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return [dict(row) for row in results]
    
    def delete_result(self, result_id: int):
        """결과 삭제"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM analysis_results WHERE id = %s', (result_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def get_count(self) -> int:
        """전체 결과 개수"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM analysis_results')
        count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return count
    
    def get_analyzed_companies(self) -> List[str]:
        """분석 완료된 종목명 목록"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT company_name FROM analysis_results')
        companies = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return companies
    
    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame 변환 (엑셀 다운로드용)"""
        conn = self.get_connection()
        df = pd.read_sql_query('SELECT * FROM analysis_results ORDER BY created_at DESC', conn)
        conn.close()
        return df
