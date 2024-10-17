% Görüntüyü yükle
img = imread('.png'); % Burada kendi görüntü dosyanızın adını yazın

% Ölçekleme faktörleri
scale_factor = 0.25; % Görüntüyü %200 oranında ölçeklendirme

% Orijinal boyutları al
[original_height, original_width, channels] = size(img);

% Yeni boyutları hesapla
new_height = round(original_height * scale_factor);
new_width = round(original_width * scale_factor);

% Yeni görüntüyü oluştur
scaled_img = zeros(new_height, new_width, channels, 'uint8');

% En yakın komşu enterpolasyonu ile ölçeklendirme
for i = 1:new_height
    for j = 1:new_width
        % Orijinal görüntüdeki karşılık gelen pikselin konumunu bul
        orig_i = round(i / scale_factor);
        orig_j = round(j / scale_factor);
        
        % Sınır kontrolü
        orig_i = min(max(orig_i, 1), original_height);
        orig_j = min(max(orig_j, 1), original_width);
        
        % Yeni görüntüye piksel değerini ata
        scaled_img(i, j, :) = img(orig_i, orig_j, :);
    end
end

% Sonuçları görselleştir
figure;
subplot(1, 2, 1);
imshow(img);
title('Orijinal Görüntü');

subplot(1, 2, 2);
imshow(scaled_img);
title('En Yakın Komşu ile Ölçeklendirilmiş Görüntü');
