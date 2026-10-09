const body = pm.response.json();
const data = body.data || body;

if (data.access_token) {
    pm.environment.set("access_token", data.access_token);
}

if (data.refresh_token) {
    pm.environment.set("refresh_token", data.refresh_token);
}

const accessToken = pm.environment.get("access_token");
const refreshToken = pm.environment.get("refresh_token");
const baseUrl = pm.environment.get("base_url");

function decodeJwt(token) {
    try {
        const part = token.split(".")[1]
            .replace(/-/g, "+")
            .replace(/_/g, "/");

        const padded = part.padEnd(
            part.length + (4 - part.length % 4) % 4,
            "="
        );

        return JSON.parse(atob(padded));
    } catch (e) {
        return null;
    }
}

function applyToken(token) {
    pm.request.headers.upsert({
        key: "Authorization",
        value: `Bearer ${token}`
    });
}

if (accessToken) {
    const payload = decodeJwt(accessToken);
    const now = Math.floor(Date.now() / 1000);

    // Refresh trước khi hết hạn 30 giây
    const shouldRefresh =
        payload &&
        payload.exp &&
        payload.exp <= now + 30;

    if (!shouldRefresh) {
        applyToken(accessToken);
    } else if (refreshToken) {
        pm.sendRequest({
            url: `${baseUrl}/ewallet3/ewallet-business-agent/public-api/v1/app/user/refresh-token`,
            method: "POST",
            header: {
                "Content-Type": "application/json"
            },
            body: {
                mode: "raw",
                raw: JSON.stringify({
                    refreshToken: refreshToken
                })
            }
        }, function (error, response) {
            if (error) {
                console.error("Refresh token failed:", error);
                return;
            }

            const body = response.json();
            const data = body.data || body;

            if (!data.access_token) {
                console.error("Refresh response:", body);
                return;
            }

            pm.environment.set("access_token", data.access_token);
            applyToken(data.access_token);

            // Một số hệ thống rotate refresh token
            if (data.refresh_token) {
                pm.environment.set(
                    "refresh_token",
                    data.refresh_token
                );
            }
        });
    }
}