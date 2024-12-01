# import 
import os
import flask            # Flask
import mysql.connector  # MySQL
import hashlib          # 로그인 비밀번호 SHA2 해시화
import MSRM             # 악보 인식 모듈
from werkzeug.utils import secure_filename

# Flask 인스턴스 생성
app = flask.Flask(__name__)

# Flask 인스턴스 설정
app.secret_key = 'The_most_valuable_lessons_from_where_theory_and_reality_meet'  # 세션 데이터 보호를 위한 암호화 키
app.config['UPLOAD_FOLDER'] = './static/uploads'

# 홈 페이지
@app.route('/', methods=['GET', 'POST'])
def home():
    # 로그인 했는지 확인
    if 'username' not in flask.session: # 세션에 사용자 정보가 존재하는지 확인
        return flask.redirect(flask.url_for('login'))
    
    # 분석된 악보가 있는 경우
    if ('img_old' in flask.session) and ('img_new' in flask.session):
        return flask.render_template(
            'home.html', 
            username=flask.session['username'], 
            img_old=flask.session['img_old'],
            img_new=flask.session['img_new']
        )
    
    # 기본 홈페이지
    else:
        return flask.render_template(
            'home.html', 
            username=flask.session['username'], 
            img_old='/static/images/default_original.png',
            img_new='/static/images/default_processing.png'
        )

# AI 악보 분석 처리
@app.route('/ai', methods=['POST'])
def ai():
    # 파일 저장
    if 'file' in flask.request.files:
        file = flask.request.files['file']
        if file.filename != '':
            fileName = secure_filename(file.filename)   # 안전한 파일이름으로 저장
            filePath = os.path.join(app.config['UPLOAD_FOLDER'], fileName)
            try:
                file.save(filePath)
            except Exception as e:
                return f"파일 저장 중 오류 발생: {str(e)}", 500
    
    # 이미지 주소 세션에 저장
    flask.session['img_old'] = filePath

    # 악보(이미지) 분석
    model = MSRM.SymbolDetector(
        [filePath],             # 이미지(악보) 경로들
        note_confidences=0.7,   # 확률이 note_confidences가 넘는 이미지들만 추출
        note_iou_threshold=0.3  # 동일한 위치의 악상기호가 겹치는 비율이 note_iou_threshold이상이면 제거 작업
    )
    images, datas = model.detact(preview=False)
    for image, data in zip(images, datas):
        MSRM.save(
            image,                      # 이미지
            os.path.splitext(filePath)[0] + '_out.png', # 저장할 이름
            data[0], data[1], data[2],  # 바운딩 박스 데이터: boxes, confidence, class
            class_names=[               # 클래스별 이름들   
                'staff',
                'clef',
                'key',
                'measure',
                'rest',
                'time',
                'note', 
                'accidental',
                'articulation',
                'dynamic',
                'glissando',
                'octave',
                'ornament',
                'repetition',
                'tie'
            ],            
        )

    # 분석 처리된 이미지 주소 세션에 저장
    flask.session['img_new'] = os.path.splitext(filePath)[0] + '_out.png'

    # home 페이지로 리디렉션
    return flask.redirect(flask.url_for('home'))

# 로그인 페이지
@app.route('/login', methods=['GET', 'POST'])
def login():
    # 로그인 했는지 확인
    if 'username' in flask.session: # 세션에 사용자 정보가 존재하는지 확인
        return flask.redirect(flask.url_for('home'))
    
    # 폼을 제출한 경우
    if flask.request.method == 'POST':
        # 사용자가 제출한 폼 읽기
        username = flask.request.form['username']  # 폼에서 전달된 사용자 이름
        password = flask.request.form['password']  # 폼에서 전달된 비밀번호
        
        # 변수 초기화
        db = None
        cursor = None

        try:
            # MySQL 데이터베이스 연결
            db = mysql.connector.connect(
                host="35.225.0.174",     # MySQL 서버 주소
                user="root",             # MySQL 사용자 이름
                password="000219",       # MySQL 비밀번호
                database="login_system", # 사용하려는 데이터베이스 이름
                connect_timeout=2        # 연결 시간 제한 (2초)
            )

            # MySQL 커서 생성
            cursor = db.cursor(dictionary=True)
             
            # 사용자 이름에 기반하여 사용자 조회
            query = "SELECT * FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            user = cursor.fetchone()  # 첫 번째 결과 가져오기

            # 사용자가 존재한다면
            if user:
                # 입력한 비밀번호를 SHA2로 해시화
                hashed_password = hashlib.sha256(password.encode()).hexdigest()
                
                # 비밀번호 일치 확인
                if user['password'] == hashed_password:  # 비밀번호 비교
                    # 로그인 성공 : 세션에 사용자 정보 저장
                    flask.session['username'] = user['username']
                    # 로그인 성공 : 세션에 사용자 정보 저장
                    return flask.redirect(flask.url_for('home'))
                else:
                    # 로그인 실패 : 비밀번호 불일치.
                    return flask.render_template('login.html', message='비밀번호를 다시 확인하여 주세요.')
            else:
                # 로그인 실패 : 존재하지 않는 사용자.
                return flask.render_template('login.html', message='존재하지 않는 사용자 이름입니다.')

        except mysql.connector.Error as e:
            # MySQL 연결 에러 처리
            return flask.render_template('login.html', message='데이터 베이스 연결에 문제가 있습니다.')
        
        finally:
            # MySQL 연결 종료
            if cursor:  cursor.close()
            if db:      db.close()
    
    # 기본 로그인 화면
    return flask.render_template('login.html', message='로그인하여 주세요.')

# 로그아웃 실행
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    # 세션 데이터 모두 삭제
    flask.session.clear()
    # 세션에서 사용자 정보 제거
    #flask.session.pop('username', None)  

    # 로그인 페이지로 리디렉션
    return flask.redirect(flask.url_for('login'))

# 애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True)
