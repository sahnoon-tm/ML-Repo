import psycopg2

def insert_complaint(text, category):
    try:
        conn = psycopg2.connect(
            dbname="complaint_system",
            user="sahnoontm",         
            password="7526",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()
        query = """
            INSERT INTO complaints (complaint_text, predicted_category)
            VALUES (%s, %s)
        """
        cursor.execute(query, (text, category))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print("Database insert error:", e)
        return False
