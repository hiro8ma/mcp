-- アプリ接続用の非特権ロール。
--
-- RLS は superuser とテーブル所有者をバイパスするため、
-- このロールで接続しない限りテナント分離は機能しない。
-- schema.sql を流したあとに、所有者（postgres）で実行する。
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user LOGIN PASSWORD 'lab';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON items TO app_user;
